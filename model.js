// V4.1-Flash attention —— 结构常量与形状推导。
//
// CFG 的每一项都来自 checkpoint 的 config.json(text_config)，字段名原样保留，
// 没有经过换算。推导逻辑（哪一维被 TP 切、O_A 的分组宽度怎么来）来自实现源码，
// 见 README 的"来源"一节。本文件不含任何实测时间。

export const CFG = Object.freeze({
  hidden_size: 5120,
  num_hidden_layers: 40,
  num_attention_heads: 64,
  num_key_value_heads: 1,
  head_dim: 512,
  qk_rope_head_dim: 64,
  q_lora_rank: 1280,
  o_lora_rank: 1024,
  o_groups: 8,
  index_n_heads: 32,
  index_head_dim: 128,
  index_topk: 512,
  candidate_block_size: 8,
  candidate_topk_blocks: 2048,
  candidate_source_layer_id: 20,
  engram_n_heads: 8,
  engram_head_dim: 256,
  sliding_window: 128,
  max_position_embeddings: 1048576,
  n_routed_experts: 384,
  num_experts_per_tok: 6,
  dspark_block_size: 5,
});

// QKV_A 的输出宽度：q 潜向量 + KV 潜向量。KV 只有 1 个头，宽度就是 head_dim。
export const KV_LATENT = CFG.num_key_value_heads * CFG.head_dim;   // 512
export const QKV_A_OUT = CFG.q_lora_rank + KV_LATENT;              // 1792

export const TILE = 128;    // 输出分块边长，即一个 tile 的边长
export const SMS = 152;     // GB200 单卡 SM 数

// 一次 FP8 block-scaled GEMM 的 scale 开销：沿 K 每 32 个元素一个 UE8M0。
const FP8_SCALE_GROUP = 32;

export const DTYPE_BYTES = { fp8: 1, bf16: 2 };

/**
 * 给定每卡 query 行数 M 和 attention 并行度 tp，返回四个投影的完整形状。
 * tp = 1 即 Attention DP（本卡持有全部 64 个头）。
 */
export function projections(M, tp) {
  const g = CFG.o_groups / tp;                       // 本卡持有的 O_A 分组数
  const qbN = (CFG.num_attention_heads * CFG.head_dim) / tp;
  return [
    {
      id: 'QKV_A', label: 'QKV_A', dtype: 'fp8', groups: 1,
      M, K: CFG.hidden_size, N: QKV_A_OUT,
      shard: null,
      globalN: QKV_A_OUT, globalK: CFG.hidden_size,
      expr: `[M, ${CFG.hidden_size}] × [${CFG.hidden_size}, ${QKV_A_OUT}] → [M, ${QKV_A_OUT}]`,
      why: `输出 ${QKV_A_OUT} = q 潜向量 ${CFG.q_lora_rank} + KV 潜向量 ${KV_LATENT}。` +
           `整张权重在每张卡上都有一份（replicated），不随 TP 变。`,
    },
    {
      id: 'Q_B', label: 'Q_B', dtype: 'fp8', groups: 1,
      M, K: CFG.q_lora_rank, N: qbN,
      shard: tp > 1 ? 'N' : null,
      globalN: CFG.num_attention_heads * CFG.head_dim, globalK: CFG.q_lora_rank,
      expr: `[M, ${CFG.q_lora_rank}] × [${CFG.q_lora_rank}, ${qbN}] → [M, ${qbN}] ` +
            `→ 视作 [M, ${CFG.num_attention_heads / tp}, ${CFG.head_dim}]`,
      why: `把 ${CFG.q_lora_rank} 维的 q 潜向量展开成 ${CFG.num_attention_heads} 个头 × ${CFG.head_dim}。` +
           `按列切（column-parallel），本卡拿 ${CFG.num_attention_heads / tp} 个头。` +
           `K 只有 ${CFG.q_lora_rank}，是四个里最浅的。`,
    },
    {
      id: 'O_A', label: 'O_A', dtype: 'bf16', groups: g,
      M, K: (CFG.num_attention_heads * CFG.head_dim) / CFG.o_groups, N: CFG.o_lora_rank * g,
      perGroupN: CFG.o_lora_rank,
      shard: tp > 1 ? 'N' : null,
      globalN: CFG.o_lora_rank * CFG.o_groups,
      globalK: (CFG.num_attention_heads * CFG.head_dim) / CFG.o_groups,
      expr: `BMM：[M, ${g}, 4096] × [${g}, 4096, ${CFG.o_lora_rank}] → [M, ${g}, ${CFG.o_lora_rank}]`,
      why: `${CFG.o_groups} 个头一组，每组输入宽度 ${CFG.num_attention_heads}×${CFG.head_dim}÷${CFG.o_groups} = ` +
           `${(CFG.num_attention_heads * CFG.head_dim) / CFG.o_groups}。` +
           `全局 ${CFG.o_groups} 组，本卡 ${g} 组。这是一个 batched GEMM，不是一个整块。`,
    },
    {
      id: 'O_B', label: 'O_B', dtype: 'fp8', groups: 1,
      M, K: (CFG.o_lora_rank * CFG.o_groups) / tp, N: CFG.hidden_size,
      shard: tp > 1 ? 'K' : null,
      globalN: CFG.hidden_size, globalK: CFG.o_lora_rank * CFG.o_groups,
      expr: `[M, ${(CFG.o_lora_rank * CFG.o_groups) / tp}] × ` +
            `[${(CFG.o_lora_rank * CFG.o_groups) / tp}, ${CFG.hidden_size}] → [M, ${CFG.hidden_size}]`,
      why: `全局 K = ${CFG.o_groups}×${CFG.o_lora_rank} = ${CFG.o_lora_rank * CFG.o_groups}，` +
           `按行切（row-parallel），切的是 K 不是 N，所以块的列数不随 TP 变，` +
           `但每卡算完要 all-reduce。`,
    },
  ];
}

/** 分块与轮数（wave）：输出切成 TILE×TILE，一块派一个 SM。O_A 按分组分别切。 */
export function tiling(p) {
  const rows = Math.ceil(p.M / TILE);
  const colsPerGroup = Math.ceil((p.perGroupN ?? p.N) / TILE);
  const cols = colsPerGroup * (p.perGroupN ? p.groups : 1);
  const tiles = cols * rows;
  const waves = Math.ceil(tiles / SMS);
  const lastWave = tiles - (waves - 1) * SMS;
  return { rows, cols, colsPerGroup, tiles, waves, lastWave, occupancy: lastWave / SMS };
}

/** 计算量与字节。都是形状算术，没有实测成分。 */
export function cost(p) {
  const batch = p.perGroupN ? p.groups : 1;   // O_A 是 BMM，输入是 [M, groups, K]
  const flops = 2 * p.M * p.K * p.N;
  const wBytes = p.K * p.N * DTYPE_BYTES[p.dtype] +
    (p.dtype === 'fp8' ? (p.K * p.N) / FP8_SCALE_GROUP : 0);
  const actIn = p.M * p.K * batch * DTYPE_BYTES[p.dtype];
  const actOut = p.M * p.N * DTYPE_BYTES.bf16;
  return { flops, wBytes, actIn, actOut, batch, arithIntensity: flops / (wBytes + actIn + actOut) };
}

/** 稀疏注意力的两级漏斗。sparse_stage = 2。 */
export function sparseFunnel(context) {
  const candidates = Math.min(
    context, CFG.candidate_topk_blocks * CFG.candidate_block_size);
  const selected = Math.min(candidates, CFG.index_topk);
  return {
    context,
    candidates,
    selected,
    blocks: Math.ceil(candidates / CFG.candidate_block_size),
    keepRatio: selected / context,
  };
}

/** core attention 本身的计算量（MLA：K 和 V 共用同一份 512 维潜向量）。 */
export function attnCost(M, selected, tp) {
  const heads = CFG.num_attention_heads / tp;
  const qk = 2 * M * heads * selected * CFG.head_dim;
  const pv = 2 * M * heads * selected * CFG.head_dim;
  const idx = 2 * M * CFG.index_n_heads * CFG.index_head_dim;
  return { heads, qk, pv, total: qk + pv, indexerPerCandidate: idx };
}

/** 层的分类：压缩率、全局 KV 生成层、重打分层、Engram 层。 */
export const LAYER_CLASSES = Object.freeze({
  swaOnly: [0, 1],
  compress2: Array.from({ length: 18 }, (_, i) => i + 2),   // 2..19
  compress1: Array.from({ length: 20 }, (_, i) => i + 20),  // 20..39
  kvSource: [2, 8, 14, 20],
  reindex: [24, 28, 32, 36],
  engram: [1, 14],
});

export const fmt = (n) => n.toLocaleString('en-US');

export function si(n, unit = '') {
  const t = [[1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'K']];
  for (const [k, s] of t) if (Math.abs(n) >= k) return (n / k).toFixed(2) + ' ' + s + unit;
  return n.toFixed(0) + ' ' + unit;
}
