// dots3 attention —— 结构常量与形状推导。
//
// CFG 的每一项都来自 dots3 checkpoint 的 config.json，字段名原样保留，没有换算。
// 推导逻辑（fused QKV_A 的宽度怎么拼、decode 走吸收路径时哪些 GEMM 真的在跑、
// 哪一维被 TP 切）来自实现源码 dots3_common/modeling.py 与 dsa/dsa_indexer.py，
// 见 README 的"来源"一节。本文件不含任何实测时间。

export const CFG = Object.freeze({
  hidden_size: 5120,
  num_hidden_layers: 46,
  first_k_dense_replace: 1,
  // full_attention 层
  num_attention_heads: 128,
  num_key_value_heads: 128,
  q_lora_rank: 1024,
  kv_lora_rank: 512,
  qk_nope_head_dim: 128,
  qk_rope_head_dim: 64,
  v_head_dim: 128,
  attention_gate_type: 'headwise',
  index_n_heads: 64,
  index_head_dim: 128,
  index_topk: 2048,
  // sliding_attention 层
  swa_num_attention_heads: 64,
  swa_num_key_value_heads: 64,
  swa_q_lora_rank: 1024,
  swa_kv_lora_rank: 1024,
  swa_qk_nope_head_dim: 192,
  swa_qk_rope_head_dim: 64,
  swa_v_head_dim: 128,
  swa_attention_gate_type: 'headwise',
  sliding_window_size: 512,
  max_position_embeddings: 393216,
  torch_dtype: 'bfloat16',
  n_routed_experts: 256,
  num_experts_per_tok: 8,
  n_shared_experts: 1,
  moe_intermediate_size: 1536,
});

// config.json 的 layer_types 原样：13 个 full_attention，33 个 sliding_attention。
export const LAYER_TYPES = Object.freeze([
  'full', 'full', 'swa', 'swa', 'swa', 'full', 'swa', 'swa', 'swa', 'full',
  'swa', 'swa', 'swa', 'full', 'swa', 'swa', 'swa', 'full', 'swa', 'swa',
  'swa', 'full', 'swa', 'swa', 'swa', 'full', 'swa', 'swa', 'swa', 'full',
  'swa', 'swa', 'swa', 'full', 'swa', 'swa', 'swa', 'full', 'swa', 'swa',
  'swa', 'full', 'swa', 'swa', 'swa', 'full',
]);
export const FULL_LAYERS = LAYER_TYPES.map((t, i) => (t === 'full' ? i : -1)).filter((i) => i >= 0);
export const SWA_LAYERS = LAYER_TYPES.map((t, i) => (t === 'swa' ? i : -1)).filter((i) => i >= 0);

/** 按层类型取出这一层 attention 用的那组字段（对应源码 Dots3AttentionMLAConfig.from_config）。 */
export function attnCfg(layer) {
  return layer === 'swa'
    ? {
        layer: 'swa', heads: CFG.swa_num_attention_heads, qLora: CFG.swa_q_lora_rank,
        kvLora: CFG.swa_kv_lora_rank, nope: CFG.swa_qk_nope_head_dim, rope: CFG.swa_qk_rope_head_dim,
        v: CFG.swa_v_head_dim, window: CFG.sliding_window_size - 1, indexer: false,
      }
    : {
        layer: 'full', heads: CFG.num_attention_heads, qLora: CFG.q_lora_rank,
        kvLora: CFG.kv_lora_rank, nope: CFG.qk_nope_head_dim, rope: CFG.qk_rope_head_dim,
        v: CFG.v_head_dim, window: null, indexer: true,
      };
}

export const TILE = 128;    // 输出分块边长
export const SMS = 152;     // GB200 单卡 SM 数

// BF16 checkpoint，没有 quantization_config：scale_block_n = 1，rope 维不补齐；
// gate 列按源码注释对齐到 8（DeepGEMM 要求）。
const SCALE_BLOCK_N = 1;
const GATE_ALIGN = Math.max(8, SCALE_BLOCK_N);
const ceilAlign = (x, a) => Math.ceil(x / a) * a;

export const DTYPE_BYTES = { fp8: 1, bf16: 2 };
const DTYPE = 'bf16';

/** fused_qkv_a_g_proj_with_mqa 的输出宽度：q 潜向量 + KV 潜向量 + rope + gate(本卡头数，对齐到 8)。 */
export function qkvAOut(layer, tp) {
  const a = attnCfg(layer);
  const gate = ceilAlign(a.heads / tp, GATE_ALIGN);   // headwise gate：每个本地头 1 列
  const ropePad = ceilAlign(a.rope, SCALE_BLOCK_N);
  return { total: a.qLora + a.kvLora + ropePad + gate, gate, ropePad, latent: a.kvLora + a.rope };
}

/**
 * 给定每卡 query 行数 M、attention 并行度 tp 和层类型，返回 decode（吸收路径）里
 * 真正执行的各步矩阵乘法。tp = 1 即 Attention DP。
 */
export function projections(M, tp, layer = 'full') {
  const a = attnCfg(layer);
  const h = a.heads / tp;                            // 本卡头数
  const qkvA = qkvAOut(layer, tp);
  const qbN = h * (a.nope + a.rope);
  const list = [
    {
      id: 'QKV_A', label: 'QKV_A', dtype: DTYPE, groups: 1,
      M, K: CFG.hidden_size, N: qkvA.total,
      shard: null,
      globalN: qkvA.total, globalK: CFG.hidden_size,
      expr: `[M, ${CFG.hidden_size}] × [${CFG.hidden_size}, ${qkvA.total}] → [M, ${qkvA.total}]`,
      why: `输出 ${qkvA.total} = q 潜向量 ${a.qLora} + KV 潜向量 ${a.kvLora} + rope ${qkvA.ropePad} + gate ${qkvA.gate}` +
           `（headwise gate 每个本地头 1 列，对齐到 8）。` +
           `源码里是一个 ReplicatedLinear（fused_qkv_a_g_proj_with_mqa），整张权重每卡一份；` +
           `只有 gate 这几列随本卡头数变。`,
    },
    {
      id: 'Q_B', label: 'Q_B', dtype: DTYPE, groups: 1,
      M, K: a.qLora, N: qbN,
      shard: tp > 1 ? 'N' : null,
      globalN: a.heads * (a.nope + a.rope), globalK: a.qLora,
      expr: `[M, ${a.qLora}] × [${a.qLora}, ${qbN}] → [M, ${qbN}] → 视作 [M, ${h}, ${a.nope + a.rope}]`,
      why: `把 ${a.qLora} 维的 q 潜向量展开成 ${a.heads} 个头 × (nope ${a.nope} + rope ${a.rope})。` +
           `按列切（column-parallel），本卡拿 ${h} 个头。`,
    },
  ];
  if (a.indexer) {
    const idxN = CFG.index_n_heads * CFG.index_head_dim;
    list.push({
      id: 'IDX_Q', label: 'indexer wq_b', dtype: DTYPE, groups: 1,
      M, K: a.qLora, N: idxN,
      shard: null,
      globalN: idxN, globalK: a.qLora,
      expr: `[M, ${a.qLora}] × [${a.qLora}, ${idxN}] → [M, ${idxN}] → 视作 [M, ${CFG.index_n_heads}, ${CFG.index_head_dim}]`,
      why: `indexer 自己的 query：${CFG.index_n_heads} 头 × ${CFG.index_head_dim}。` +
           `ReplicatedLinear，不随 TP 切。另有 wk（${CFG.hidden_size} → ${CFG.index_head_dim}）和 ` +
           `weights_proj（${CFG.hidden_size} → ${CFG.index_n_heads}），很窄，这里不单独画。`,
    });
  }
  list.push(
    {
      id: 'W_KC', label: 'W_KC（吸收）', dtype: DTYPE, groups: h,
      M, K: a.nope, N: a.kvLora * h, perGroupN: a.kvLora,
      shard: tp > 1 ? 'N' : null,
      globalN: a.kvLora * a.heads, globalK: a.nope,
      expr: `BMM：[${h}, M, ${a.nope}] × [${h}, ${a.nope}, ${a.kvLora}] → [M, ${h}, ${a.kvLora}]`,
      why: `decode 走吸收路径：不展开 K，而是把每个头的 q_nope 乘上 kv_b 里的 K 部分（w_kc），` +
           `投到 ${a.kvLora} 维潜空间去和 KV 潜向量直接点积。每头一个 ${a.nope}×${a.kvLora} 的小矩阵，` +
           `本卡 ${h} 个头就是 ${h} 组的 batched GEMM。`,
    },
    {
      id: 'W_VC', label: 'W_VC（吸收）', dtype: DTYPE, groups: h,
      M, K: a.kvLora, N: a.v * h, perGroupN: a.v,
      shard: tp > 1 ? 'N' : null,
      globalN: a.v * a.heads, globalK: a.kvLora,
      expr: `BMM：[${h}, M, ${a.kvLora}] × [${h}, ${a.kvLora}, ${a.v}] → [M, ${h}, ${a.v}]`,
      why: `attention 的输出还在 ${a.kvLora} 维潜空间，再用 kv_b 里的 V 部分（w_vc）每头还原成 ${a.v} 维。` +
           `同样是 ${h} 组的 batched GEMM。`,
    },
    {
      id: 'O', label: 'O', dtype: DTYPE, groups: 1,
      M, K: h * a.v, N: CFG.hidden_size,
      shard: tp > 1 ? 'K' : null,
      globalN: CFG.hidden_size, globalK: a.heads * a.v,
      expr: `[M, ${h * a.v}] × [${h * a.v}, ${CFG.hidden_size}] → [M, ${CFG.hidden_size}]`,
      why: `全局 K = ${a.heads}×${a.v} = ${a.heads * a.v}，按行切（row-parallel），切的是 K 不是 N，` +
           `所以块的列数不随 TP 变，但每卡算完要 all-reduce。dots3 的 O 是一个整块，没有 V4.1 那种 O_A/O_B 低秩分解。`,
    },
  );
  return list;
}

/** 分块与轮数（wave）：输出切成 TILE×TILE，一块派一个 SM。BMM 按分组分别切。 */
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
  const batch = p.perGroupN ? p.groups : 1;
  const flops = 2 * p.M * p.K * p.N;
  const wBytes = p.K * p.N * DTYPE_BYTES[p.dtype];
  const actIn = p.M * p.K * batch * DTYPE_BYTES[p.dtype];
  const actOut = p.M * p.N * DTYPE_BYTES.bf16;
  return { flops, wBytes, actIn, actOut, batch, arithIntensity: flops / (wBytes + actIn + actOut) };
}

/** 每行实际看多少 token：full 层由 indexer 选 top-k（单级，没有候选块那一级）；swa 层看窗口。 */
export function attnScope(context, layer = 'full') {
  const a = attnCfg(layer);
  if (layer === 'swa') {
    const selected = Math.min(context, a.window);
    return { layer, context, scanned: selected, selected, keepRatio: selected / context };
  }
  const selected = Math.min(context, CFG.index_topk);
  return { layer, context, scanned: context, selected, keepRatio: selected / context };
}

/** core attention 本身的计算量（吸收后 Q·K 在 kv_lora+rope 维，P·V 在 kv_lora 维）；以及 indexer 打分量。 */
export function attnCost(M, context, tp, layer = 'full') {
  const a = attnCfg(layer);
  const s = attnScope(context, layer);
  const heads = a.heads / tp;
  const qk = 2 * M * heads * s.selected * (a.kvLora + a.rope);
  const pv = 2 * M * heads * s.selected * a.kvLora;
  const idx = a.indexer ? 2 * M * CFG.index_n_heads * CFG.index_head_dim * s.scanned : 0;
  return { heads, qk, pv, total: qk + pv, indexer: idx, selected: s.selected, scanned: s.scanned };
}

/** 每 token 每层 KV 缓存元素数：attn_mqa 的 head_dim = kv_lora_rank + qk_rope_head_dim，单头。 */
export function kvPerToken() {
  const full = attnCfg('full'), swa = attnCfg('swa');
  const fullElems = full.kvLora + full.rope;   // 576
  const swaElems = swa.kvLora + swa.rope;      // 1088
  return {
    fullElems, swaElems,
    fullLayers: FULL_LAYERS.length, swaLayers: SWA_LAYERS.length,
    totalElems: fullElems * FULL_LAYERS.length + swaElems * SWA_LAYERS.length,
    persistElems: fullElems * FULL_LAYERS.length,   // swa 层只需保留最近 window 个 token
  };
}

export const fmt = (n) => n.toLocaleString('en-US');

export function si(n, unit = '') {
  const t = [[1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'K']];
  for (const [k, s] of t) if (Math.abs(n) >= k) return (n / k).toFixed(2) + ' ' + s + unit;
  return n.toFixed(0) + ' ' + unit;
}
