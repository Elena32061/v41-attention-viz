// 校验 kernels 页的两份数据集：结构、关键路径闭合、页面假设的字段都在。
import { readFileSync } from 'node:fs';

let fail = 0;
const ck = (name, cond, detail = '') => {
  if (cond) console.log(`ok   ${name}${detail ? ' = ' + detail : ''}`);
  else { console.log(`FAIL ${name}${detail ? ': ' + detail : ''}`); fail++; }
};
const near = (a, b, eps) => Math.abs(a - b) < eps;

// kernels.js 里的配色/中文名表，确认没有 bucket 掉进 other
const JS = readFileSync('./kernels.js', 'utf8');
const block = (name) => {
  const i = JS.indexOf(`const ${name} = {`);
  if (i < 0) throw new Error(`${name} 不在 kernels.js 里`);
  return JS.slice(i, JS.indexOf('};', i));
};
const known = new Set([...block('COLORS').matchAll(/([a-z_]+): '#/g)].map((m) => m[1]));
const named = new Set([...block('BUCKET_CN').matchAll(/([a-z_]+): '/g)].map((m) => m[1]));

const FWS = ['SGLang', 'DeepInfer'];
const RANKS = ['node_0/rank0', 'node_1/rank4'];

for (const [ds, file, wantCrit] of [
  ['aligned', './kernels_b64_8card_aligned.json', true],
  ['legacy', './kernels_b64_8card.json', false],
]) {
  console.log(`\n--- ${ds}`);
  const d = JSON.parse(readFileSync(file, 'utf8'));
  ck(`${ds} steps`, d.steps === 100, `${d.steps}`);
  ck(`${ds} has_critical_path`, (d.has_critical_path === true) === wantCrit, `${!!d.has_critical_path}`);
  ck(`${ds} limits 非空`, Array.isArray(d.limits) && d.limits.length > 0, `${d.limits.length} 条`);

  for (const f of FWS) {
    const fw = d.frameworks[f];
    ck(`${ds} ${f} run_id`, typeof fw.run_id === 'string' && fw.run_id.length > 0, fw.run_id);
    if (wantCrit) ck(`${ds} ${f} tpot_off_ms`, fw.tpot_off_ms > 0, fw.tpot_off_ms.toFixed(3));
    for (const rk of RANKS) {
      const r = fw.ranks[rk];
      ck(`${ds} ${f} ${rk} 存在`, !!r);
      for (const k of ['kernels_min', 'kernels_median', 'kernels_max', 'identical_sequence_steps',
        'identical_multiset_steps', 'modal_in_graph', 'modal_outside_graph', 'varying_kernels', 'step_ms_mean']) {
        if (r[k] === undefined) { console.log(`FAIL ${ds} ${f} ${rk} 缺字段 ${k}`); fail++; }
      }
      ck(`${ds} ${f} ${rk} min<=median<=max`, r.kernels_min <= r.kernels_median && r.kernels_median <= r.kernels_max,
        `${r.kernels_min}/${r.kernels_median}/${r.kernels_max}`);
    }
    // rank 0 带表
    const r = fw.ranks['node_0/rank0'];
    ck(`${ds} ${f} table 行数 = distinct_names`, r.table.length === r.distinct_names, `${r.table.length}`);
    ck(`${ds} ${f} modal_order 合计 = modal_kernels`,
      r.modal_order.reduce((a, [, n]) => a + n, 0) === r.modal_kernels, `${r.modal_kernels}`);
    ck(`${ds} ${f} 累计合计对得上`,
      near(r.table.reduce((s, x) => s + x.ms_per_step, 0), r.total_ms_per_step_cumulative, 1e-6),
      r.total_ms_per_step_cumulative.toFixed(4));
    for (const x of r.table) {
      if (!(x.in_graph_frac >= 0 && x.in_graph_frac <= 1)) { console.log(`FAIL in_graph_frac ${x.short}`); fail++; }
      if (!(x.count_per_step > 0)) { console.log(`FAIL count_per_step ${x.short}`); fail++; }
    }
    const buckets = [...new Set(r.table.map((x) => x.bucket))];
    const unstyled = buckets.filter((b) => !known.has(b));
    const untranslated = buckets.filter((b) => !named.has(b));
    ck(`${ds} ${f} 所有 bucket 有配色`, unstyled.length === 0, unstyled.join(',') || buckets.length + ' 类');
    ck(`${ds} ${f} 所有 bucket 有中文名`, untranslated.length === 0, untranslated.join(',') || 'ok');

    if (!wantCrit) continue;
    // 关键路径闭合：逐 kernel 关键路径 + 空隙 == 步长
    for (const rk of RANKS) {
      const q = fw.ranks[rk];
      const gaps = Object.values(q.critical_gaps_ms).reduce((s, v) => s + v, 0);
      if (rk === 'node_0/rank0') {
        const sum = q.table.reduce((s, x) => s + x.critical_ms_per_step, 0);
        ck(`${ds} ${f} ${rk} 关键路径合计对得上 table`, near(sum, q.total_critical_ms_per_step, 1e-6), sum.toFixed(4));
        ck(`${ds} ${f} ${rk} 关键路径 + 空隙 = 步长`, near(sum + gaps, q.step_ms_mean, 1e-3),
          `${(sum + gaps).toFixed(4)} vs ${q.step_ms_mean.toFixed(4)}`);
        ck(`${ds} ${f} ${rk} 关键路径 <= 累计`, sum <= q.total_ms_per_step_cumulative + 1e-9,
          `${sum.toFixed(3)} <= ${q.total_ms_per_step_cumulative.toFixed(3)}`);
      } else {
        ck(`${ds} ${f} ${rk} 有空隙记录`, gaps > 0, gaps.toFixed(4));
      }
    }
  }
}

// 老链接里的 hide 值在对齐版里仍然有效
const al = JSON.parse(readFileSync('./kernels_b64_8card_aligned.json', 'utf8'));
const allB = new Set(FWS.flatMap((f) => al.frameworks[f].ranks['node_0/rank0'].table.map((x) => x.bucket)));
for (const b of ['dense_projection', 'attention']) ck(`旧链接 hide=${b} 仍有效`, allB.has(b));

console.log(fail ? `\n${fail} FAILED` : '\n全部通过');
process.exit(fail ? 1 : 0);
