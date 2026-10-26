import { projections, tiling, cost, sparseFunnel, attnCost, CFG, QKV_A_OUT } from './model.js';

let fail = 0;
const ck = (name, got, want) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  if (!ok) { console.log(`FAIL ${name}: got ${JSON.stringify(got)} want ${JSON.stringify(want)}`); fail++; }
  else console.log(`ok   ${name} = ${JSON.stringify(got)}`);
};

ck('QKV_A_OUT', QKV_A_OUT, 1792);

// 与静态图 render.json 对照：M=896 下 DP 与 TP4 的块数/轮数必须完全一致
for (const [tp, want] of [[1, {QKV_A:[98,1], Q_B:[1792,12], O_A:[448,3], O_B:[280,2]}],
                          [4, {QKV_A:[98,1], Q_B:[448,3],  O_A:[112,1], O_B:[280,2]}]]) {
  for (const p of projections(896, tp)) {
    const t = tiling(p);
    ck(`tp${tp} ${p.id} [tiles,waves]`, [t.tiles, t.waves], want[p.id]);
  }
}
const waves = (tp) => projections(896, tp).reduce((a,p)=>a+tiling(p).waves, 0);
ck('合计轮数 DP', waves(1), 18);
ck('合计轮数 TP4', waves(4), 7);

// 形状自洽
const [qkv, qb, oa, ob] = projections(896, 1);
ck('Q_B N = 64*512', qb.N, 32768);
ck('O_A 每组 K', oa.K, 4096);
ck('O_A N = 8*1024', oa.N, 8192);
ck('O_B K = 8*1024', ob.K, 8192);
ck('TP4 Q_B N', projections(896,4)[1].N, 8192);
ck('TP4 O_B K (切K不切N)', projections(896,4)[3].K, 2048);
ck('TP4 O_B N 不变', projections(896,4)[3].N, 5120);
ck('TP4 O_A 组数', projections(896,4)[2].groups, 2);

// FLOPs：O_A 的 BMM 展开后应与整块写法相等
const oaF = cost(oa).flops;
ck('O_A FLOPs = 8 组之和', oaF, 8 * 2 * 896 * 4096 * 1024);
ck('O_A actIn 含 batch 维', cost(oa).actIn, 896 * 4096 * 8 * 2);
ck('QKV_A FP8 权重字节含 scale', cost(qkv).wBytes, 5120*1792 + 5120*1792/32);

// wave 边界：96 块宽的 QKV_A，一轮装 152 块 → rb<=10 时 1 轮（14*10=140<=152），rb=11 → 2 轮
ck('QKV_A M=1280 轮数', tiling(projections(1280,1)[0]).waves, 1);
ck('QKV_A M=1408 轮数', tiling(projections(1408,1)[0]).waves, 2);

// 稀疏漏斗
ck('16K 上下文候选封顶', sparseFunnel(16384).candidates, 16384);
ck('1M 上下文候选封顶', sparseFunnel(1048576).candidates, 2048*8);
ck('选中数', sparseFunnel(1048576).selected, 512);
ck('短上下文不虚报', sparseFunnel(256).selected, 256);
ck('attn heads DP', attnCost(896,512,1).heads, 64);

console.log(fail ? `\n${fail} FAILED` : '\n全部通过');
process.exit(fail ? 1 : 0);
