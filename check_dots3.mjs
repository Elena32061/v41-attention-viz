import { projections, tiling, cost, attnScope, attnCost, kvPerToken, qkvAOut, CFG, FULL_LAYERS, SWA_LAYERS } from './model_dots3.js';

let fail = 0;
const ck = (name, got, want) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) { console.log(`FAIL ${name}: got ${JSON.stringify(got)} want ${JSON.stringify(want)}`); fail++; }
  else console.log(`ok   ${name} = ${JSON.stringify(got)}`);
};

// 层分布与 config.json 的 layer_types 一致
ck('full 层号', FULL_LAYERS, [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]);
ck('swa 层数', SWA_LAYERS.length, 33);

// fused_qkv_a_g_proj_with_mqa 宽度 = q_lora + kv_lora + rope + ceil8(local_heads)
ck('full QKV_A DP', qkvAOut('full', 1).total, 1024 + 512 + 64 + 128);
ck('full QKV_A TP4', qkvAOut('full', 4).total, 1024 + 512 + 64 + 32);
ck('swa QKV_A DP', qkvAOut('swa', 1).total, 1024 + 1024 + 64 + 64);
ck('swa QKV_A TP8', qkvAOut('swa', 8).total, 1024 + 1024 + 64 + 8);

// 形状自洽
const [qkv, qb, idx, wkc, wvc, o] = projections(896, 1, 'full');
ck('full 步骤顺序', [qkv.id, qb.id, idx.id, wkc.id, wvc.id, o.id], ['QKV_A', 'Q_B', 'IDX_Q', 'W_KC', 'W_VC', 'O']);
ck('Q_B N = 128*(128+64)', qb.N, 24576);
ck('IDX_Q N = 64*128', idx.N, 8192);
ck('W_KC 每头 K,N', [wkc.K, wkc.perGroupN, wkc.groups], [128, 512, 128]);
ck('W_VC 每头 K,N', [wvc.K, wvc.perGroupN, wvc.groups], [512, 128, 128]);
ck('O K = 128*128', o.K, 16384);
ck('TP4 Q_B N', projections(896, 4, 'full')[1].N, 6144);
ck('TP4 O K', projections(896, 4, 'full')[5].K, 4096);
const sw = projections(896, 1, 'swa');
ck('swa 没有 indexer', sw.map((p) => p.id), ['QKV_A', 'Q_B', 'W_KC', 'W_VC', 'O']);
ck('swa Q_B N = 64*(192+64)', sw[1].N, 16384);
ck('swa O K = 64*128', sw[4].K, 8192);

// 分块：M=896 → 7 块高
for (const p of projections(896, 1, 'full')) ck(`rows ${p.id}`, tiling(p).rows, 7);
ck('full DP QKV_A tiles', tiling(qkv).tiles, 14 * 7);
ck('full DP 合计轮数', projections(896, 1, 'full').reduce((a, p) => a + tiling(p).waves, 0), 1 + 9 + 3 + 24 + 6 + 2);

// KV：与此前算的 43,392 元素/token 一致
ck('KV per token', kvPerToken().totalElems, 43392);
ck('KV persist per token', kvPerToken().persistElems, 13 * 576);

// 稀疏范围
ck('full 16K 选中', attnScope(16384, 'full').selected, 2048);
ck('full 1K 全看', attnScope(1024, 'full').selected, 1024);
ck('swa 16K 窗口', attnScope(16384, 'swa').selected, 511);
ck('indexer 扫全长', attnScope(262144, 'full').scanned, 262144);
const ac = attnCost(896, 16384, 1, 'full');
ck('core attention QK/PV 维度', [ac.qk / (2 * 896 * 128 * 2048), ac.pv / (2 * 896 * 128 * 2048)], [576, 512]);
ck('indexer FLOPs', ac.indexer, 2 * 896 * 64 * 128 * 16384);

// cost 口径：BF16 权重，无 scale
ck('QKV_A 权重字节', cost(qkv).wBytes, 5120 * 1728 * 2);

console.log(fail ? `\n${fail} FAILED` : '\nall ok');
process.exit(fail ? 1 : 0);
