const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const fmt = (n, d = 0) => n.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const COLORS = {
  moe_grouped_gemm: '#0f6ab4', dense_projection: '#c8812a', attention: '#2a9d8f', mhc: '#8e5ea2',
  quantize_norm: '#d4a017', routing_dispatch: '#e76f51', collective: '#d1495b', eager_glue: '#9aa3ab',
  sampling: '#5a9e6f', swiglu: '#6c8ebf', engram: '#b5651d', topk_unresolved: '#c06c84',
  memcpy: '#7f8c8d', memset: '#95a5a6', other: '#777',
};
const BUCKET_CN = {
  moe_grouped_gemm: 'MoE（含 EP 通信）', dense_projection: '稠密投影', attention: 'attention', mhc: 'mHC',
  quantize_norm: '量化 / norm', routing_dispatch: '路由 / gate', collective: 'NCCL 集合通信（MoE 之外）', eager_glue: '胶水小 kernel',
  sampling: '采样', swiglu: 'SwiGLU', engram: 'Engram', topk_unresolved: 'top-k',
  memcpy: 'memcpy', memset: 'memset', other: '其他',
};
const DATASETS = {
  aligned: {
    file: './kernels_b64_8card_aligned.json',
    label: '对齐版（两侧都开调度重叠）',
    blurb: '同一 Trial 2266850 的同两个节点，两侧都开调度重叠、DeepInfer 新配置。SGLang 2026-10-25，DeepInfer 2026-10-26。',
    build: 'outputs/sg_di_kernel_compare_aligned_20261026/stage/build_kernels_json.py',
  },
  legacy: {
    file: './kernels_b64_8card.json',
    label: '旧版（SGLang 关了调度重叠）',
    blurb: 'mentor 的 qs8 两节点，DeepInfer 2026-10-22、SGLang 2026-10-23，SGLang 跑时 disable_overlap_schedule=True。两份不同天、配置未对齐。',
    build: 'outputs/sg_di_8card_kernel_compare_20261025/stage/build_kernels_json.py',
  },
};
const state = { ds: 'aligned', fw: 'both', sort: 'crit', hide: new Set() };
const cache = {};
const FWS = ['SGLang', 'DeepInfer'];
let data;

async function loadDataset(ds) {
  cache[ds] ??= await (await fetch(DATASETS[ds].file)).json();
  data = cache[ds];
}
const hasCrit = () => data.has_critical_path === true;
const r0 = (fw) => data.frameworks[fw].ranks['node_0/rank0'];
const r4 = (fw) => data.frameworks[fw].ranks['node_1/rank4'];
const bucketList = () => [...new Set(FWS.flatMap((f) => r0(f).table.map((x) => x.bucket)))].sort(
  (a, b) => (COLORS[a] ? 0 : 1) - (COLORS[b] ? 0 : 1) || a.localeCompare(b));
const sortKeys = () => (hasCrit() ? ['crit', 'ms', 'count', 'us', 'bucket'] : ['ms', 'count', 'us', 'bucket']);
const SORT_CN = {
  crit: '按每步关键路径时间', ms: '按每步累计时间', count: '按每步次数',
  us: '按单次平均时长', bucket: '按功能分组',
};
const SORT_HINT = {
  crit: '该 kernel 占住这一步墙钟的时长。每 rank 的关键路径 + 空隙 = 步长。',
  ms: '累计 = 次数 × 单次时长，重叠的部分会重复计入，不是墙钟。',
  count: '一步里被启动的次数。',
  us: '单个 kernel 的平均时长。',
  bucket: '同组内再按时间排。',
};

function readURL() {
  const q = new URLSearchParams(location.search);
  if (Object.keys(DATASETS).includes(q.get('ds'))) state.ds = q.get('ds');
  if (['both', ...FWS].includes(q.get('fw'))) state.fw = q.get('fw');
  if (['crit', 'ms', 'count', 'us', 'bucket'].includes(q.get('sort'))) state.sort = q.get('sort');
  if (q.get('hide')) state.hide = new Set(q.get('hide').split(',').filter(Boolean));
}
function writeURL() {
  const q = new URLSearchParams({ ds: state.ds, fw: state.fw, sort: state.sort });
  if (state.hide.size) q.set('hide', [...state.hide].join(','));
  history.replaceState(null, '', `${location.pathname}?${q}`);
}
const shown = () => (state.fw === 'both' ? FWS : [state.fw]);
const critOf = (x) => x.critical_ms_per_step ?? 0;

function renderDatasetNote() {
  const d = DATASETS[state.ds];
  const tp = FWS.map((f) => `${f} ${data.frameworks[f].tpot_off_ms ? fmt(data.frameworks[f].tpot_off_ms, 3) + ' ms' : '—'}`).join('、');
  $('dsNote').innerHTML = `<b>${d.label}。</b>${d.blurb}` +
    (hasCrit()
      ? `对应的正常计时（profiler 关）是 ${tp}。本页所有时间都来自 profiler 开着的 trace，只作诊断。`
      : ` 这份数据只有累计时长，没有关键路径分解 —— 当时的统计没算，这里不追补。`);
}

function renderOverview() {
  const rows = [
    ['每步 kernel 数（中位 / 最小–最大）', (r) => `${r.kernels_median}（${r.kernels_min}–${r.kernels_max}）`],
    ['其中在 CUDA graph 里 / 外', (r) => `${r.modal_in_graph} / ${r.modal_outside_graph}`],
    ['100 步里 kernel 集合完全相同的步数', (r) => `${r.identical_multiset_steps}`],
    ['100 步里 kernel 顺序也完全相同的步数', (r) => `${r.identical_sequence_steps}`],
    ['不同 kernel 名数', (f) => `${r0(f).distinct_names}`, 'rank0'],
  ];
  if (hasCrit()) {
    rows.push(
      ['每步<b>关键路径</b>时长', (f) => `<b>${fmt(r0(f).total_critical_ms_per_step, 3)} ms</b>`, 'rank0'],
      ['关键路径上的空隙', (f) => {
        const g = r0(f).critical_gaps_ms || {};
        return `${fmt(Object.values(g).reduce((s, v) => s + v, 0), 3)} ms`;
      }, 'rank0']);
  }
  rows.push(
    ['每步累计 kernel 时长', (f) => `${fmt(r0(f).total_ms_per_step_cumulative, 2)} ms`, 'rank0'],
    ['profiler 开着时的步长', (r) => `${fmt(r.step_ms_mean, 3)} ms`],
    ['集合有差异的步里变化的 kernel', (r) => Object.entries(r.varying_kernels).map(([k, v]) => `${k.split('::').pop()}（${v} 步）`).join('<br>') || '无']);

  const head = `<tr><th></th>${FWS.flatMap((f) => [`<th>${f}<br><span style="font-weight:400">rank 0</span></th>`, `<th>${f}<br><span style="font-weight:400">rank 4</span></th>`]).join('')}</tr>`;
  $('overview').innerHTML = head + rows.map(([n, fn, scope]) => scope === 'rank0'
    ? `<tr><td>${n}<span style="color:var(--grey)"> · 仅 rank 0</span></td>${FWS.map((f) => `<td colspan="2">${fn(f)}</td>`).join('')}</tr>`
    : `<tr><td>${n}</td>${FWS.flatMap((f) => [`<td>${fn(r0(f), f)}</td>`, `<td>${fn(r4(f), f)}</td>`]).join('')}</tr>`).join('');
  const tp = FWS.map((f) => data.frameworks[f].tpot_off_ms);
  if (tp.every(Boolean)) {
    $('overview').innerHTML +=
      `<tr><td>同配置的正常计时 TPOT（profiler 关）</td>` +
      tp.map((v) => `<td colspan="2">${fmt(v, 3)} ms</td>`).join('') + `</tr>`;
  }

  const sg = r0('SGLang'), di = r0('DeepInfer');
  const ratio = (sg.kernels_median / di.kernels_median).toFixed(1);
  $('overviewNote').innerHTML = hasCrit()
    ? `<b>SGLang 一步 ${sg.kernels_median} 个 kernel，DeepInfer ${di.kernels_median} 个，差 ${ratio} 倍，` +
      `但关键路径只差 ${fmt(di.total_critical_ms_per_step - sg.total_critical_ms_per_step, 3)} ms。</b>` +
      `SGLang 把 mHC、norm、量化、胶水拆成大量小 kernel 并发跑，所以它的累计时长 ` +
      `${fmt(sg.total_ms_per_step_cumulative, 2)} ms 比步长 ${fmt(sg.step_ms_mean, 2)} ms 还高 —— ` +
      `重叠的部分被重复计入了。DeepInfer 几乎不重叠，累计 ${fmt(di.total_ms_per_step_cumulative, 2)} ms 和步长基本贴合。` +
      `<b>所以跨框架只能比关键路径那一列，不能比累计。</b>` +
      `两个节点上的 rank 0 和 rank 4 数字一致。`
    : `<b>每步跑的 kernel 集合都一样。</b>SGLang 100 步集合相同，但顺序每步都不同 —— 它多流并发，不同 stream 上 kernel 的启动先后每步会变；` +
      `DeepInfer 99 步顺序一模一样，剩下 1 步多了几个 reduce / elementwise 和一次 AllGather。` +
      `SGLang 一步 ${sg.kernels_median} 个 kernel，DeepInfer ${di.kernels_median} 个，差 ${ratio} 倍，` +
      `差在 SGLang 把 mHC、norm、量化、胶水拆成很多小 kernel。两个节点上的 rank 0 和 rank 4 数字一致。`;
}

function renderStrips() {
  $('legend').innerHTML = bucketList().map((b) => `<span style="--c:${COLORS[b] || COLORS.other}">${BUCKET_CN[b] || b}</span>`).join('');
  const W = 1420, H = 64;
  const html = [];
  for (const f of shown()) {
    const order = r0(f).modal_order;
    const total = order.reduce((a, [, n]) => a + n, 0);
    const n2b = Object.fromEntries(r0(f).table.map((x) => [x.short, x.bucket]));
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('preserveAspectRatio', 'none');
    let x = 0;
    for (const [name, n, inGraph] of order) {
      const w = (n / total) * W;
      const rect = document.createElementNS(NS, 'rect');
      rect.setAttribute('x', x); rect.setAttribute('y', inGraph ? 10 : 0); rect.setAttribute('width', Math.max(w, 0.4)); rect.setAttribute('height', inGraph ? 44 : 54);
      rect.setAttribute('fill', state.hide.has(n2b[name]) ? '#eceff1' : (COLORS[n2b[name]] || COLORS.other));
      const t = document.createElementNS(NS, 'title');
      t.textContent = `${name}${n > 1 ? ' ×' + n : ''}${inGraph ? '' : '（graph 外）'}`;
      rect.appendChild(t); svg.appendChild(rect); x += w;
    }
    const wrap = document.createElement('div'); wrap.className = 'strip';
    wrap.innerHTML = `<div class="fw-title">${f} <span class="mono">一步 ${total} 个 kernel，${order.length} 段（连续同名合并）；矮条 = graph 内，高条 = graph 外</span></div>`;
    wrap.appendChild(svg); html.push(wrap);
  }
  $('strips').replaceChildren(...html);
  $('stripNote').innerHTML =
    `横向是执行顺序（按 kernel 的 start 时间），宽度按 kernel <b>个数</b>均分，不按时间。` +
    `能看到的是 40 层的周期结构：每层一组"投影 → attention → 投影 → mHC → gate → 专家 GEMM"。` +
    `SGLang 每层里夹着很多灰色胶水和黄色量化；DeepInfer 每层只有十几个。悬停看名字。`;
}

function renderBuckets() {
  const crit = hasCrit();
  const html = [];
  for (const f of shown()) {
    const t = r0(f).table;
    const agg = {};
    for (const x of t) {
      const a = agg[x.bucket] ??= { ms: 0, cr: 0, n: 0, kinds: 0 };
      a.ms += x.ms_per_step; a.cr += critOf(x); a.n += x.count_per_step; a.kinds += 1;
    }
    const tot = Object.values(agg).reduce((s, a) => s + a.ms, 0);
    const totCr = Object.values(agg).reduce((s, a) => s + a.cr, 0);
    const rows = Object.entries(agg).sort((a, b) => (crit ? b[1].cr - a[1].cr : b[1].ms - a[1].ms)).map(([b, a]) =>
      `<tr><td><span class="swatch" style="background:${COLORS[b] || COLORS.other}"></span>${BUCKET_CN[b] || b}</td>` +
      `<td>${fmt(a.n, a.n % 1 ? 2 : 0)}</td><td>${a.kinds}</td>` +
      (crit ? `<td><b>${fmt(a.cr, 3)}</b></td><td>${fmt((a.cr / totCr) * 100, 1)}%</td>` : '') +
      `<td>${fmt(a.ms, 3)}</td>${crit ? '' : `<td>${fmt((a.ms / tot) * 100, 1)}%</td>`}</tr>`).join('');
    const gaps = r0(f).critical_gaps_ms || {};
    const gapMs = Object.values(gaps).reduce((s, v) => s + v, 0);
    const gapRow = crit
      ? `<tr><td style="color:var(--grey)">关键路径空隙（无 kernel 在等）</td><td></td><td></td>` +
        `<td>${fmt(gapMs, 3)}</td><td>${fmt((gapMs / r0(f).step_ms_mean) * 100, 1)}%</td><td></td></tr>` : '';
    const sec = document.createElement('div');
    sec.innerHTML = `<div class="fw-title">${f}</div><table>` +
      `<tr><th>功能</th><th>次/步</th><th>种类</th>${crit ? '<th>关键路径 ms/步</th><th>占关键路径</th>' : ''}` +
      `<th>累计 ms/步</th>${crit ? '' : '<th>占累计</th>'}</tr>${rows}${gapRow}` +
      `<tr class="total"><td>合计</td><td>${fmt(t.reduce((s, x) => s + x.count_per_step, 0), 0)}</td><td>${t.length}</td>` +
      (crit ? `<td><b>${fmt(totCr + gapMs, 3)}</b></td><td>= 步长 ${fmt(r0(f).step_ms_mean, 3)}</td>` : '') +
      `<td>${fmt(tot, 3)}</td>${crit ? '' : '<td></td>'}</tr></table>`;
    html.push(sec);
  }
  $('buckets').replaceChildren(...html);
  const commNote = `<br><b>为什么看不到 dispatch / combine：</b>两边 MoE 都走 MegaMoE（<code>deep_gemm::sm100_fp8_fp4_mega_moe_impl</code>，每层 1 个）。` +
    `EP 的 dispatch、专家 GEMM、combine 融合在这一个 kernel 里，借 NVLink 直接读写对端显存完成，不调用 NCCL，所以 EP 通信时间都算在"MoE（含 EP 通信）"这一类里，目前没有拆开。` +
    `"NCCL 集合通信"只统计 MoE 之外的 NCCL 调用（SGLang 每步 2 次 ReduceScatter、3 次 AllGather、1 次 AllReduce，每步而非每层），不是 MoE 的 dispatch / combine。` +
    `"路由 / gate"是 gate 打分、top-k 和 dispatch 前的布局准备（如 SGLang 的 mega_moe_pre_dispatch），不搬运 token。`;
  $('bucketNote').innerHTML = (crit
    ? `关键路径那一列逐行加起来，再加上空隙，正好等于这一步的墙钟长度 —— 可以直接读成"这一步的时间花在哪"。` +
      `累计列保留做参考：它是逐 kernel 相加，并发跑的部分重复计入了。`
    : `这份数据只有累计时长。SGLang 多流并发，累计值高于墙钟，两个框架之间不要直接相减。`) + commNote;
}

function renderTables() {
  const crit = hasCrit();
  const key = { crit: (x) => -critOf(x), ms: (x) => -x.ms_per_step, count: (x) => -x.count_per_step, us: (x) => -x.us_mean, bucket: () => 0 }[state.sort];
  const tie = crit ? (a, b) => critOf(b) - critOf(a) : (a, b) => b.ms_per_step - a.ms_per_step;
  const html = [];
  for (const f of shown()) {
    let t = r0(f).table.filter((x) => !state.hide.has(x.bucket));
    t = [...t].sort((a, b) => (state.sort === 'bucket' ? (a.bucket.localeCompare(b.bucket) || tie(a, b)) : key(a) - key(b)));
    const rows = t.map((x) =>
      `<tr><td class="name" title="${x.short}"><span class="swatch" style="background:${COLORS[x.bucket] || COLORS.other}"></span>${x.short}</td>` +
      `<td>${fmt(x.count_per_step, x.count_per_step % 1 ? 2 : 0)}</td>` +
      (crit ? `<td><b>${fmt(critOf(x), 4)}</b></td>` : '') +
      `<td>${fmt(x.ms_per_step, 4)}</td><td>${fmt(x.us_mean, 1)}</td>` +
      `<td>${x.in_graph_frac >= 0.999 ? '内' : x.in_graph_frac <= 0.001 ? '<b class="hl">外</b>' : fmt(x.in_graph_frac * 100, 0) + '%'}</td></tr>`).join('');
    const sec = document.createElement('div');
    sec.innerHTML = `<div class="fw-title">${f} <span class="mono">${t.length} 种</span></div><div class="scroll"><table>` +
      `<tr><th>kernel</th><th>次/步</th>${crit ? '<th>关键路径 ms/步</th>' : ''}<th>累计 ms/步</th><th>单次 μs</th><th>graph</th></tr>${rows}</table></div>`;
    html.push(sec);
  }
  $('tables').replaceChildren(...html);
  $('sortHint').textContent = SORT_HINT[state.sort];
}

function renderFilters() {
  $('filters').innerHTML = bucketList().map((b) =>
    `<button data-b="${b}" class="${state.hide.has(b) ? '' : 'on'}" style="border-left:4px solid ${COLORS[b] || COLORS.other}">${BUCKET_CN[b] || b}</button>`).join('');
  for (const btn of $('filters').querySelectorAll('button')) {
    btn.addEventListener('click', () => { const b = btn.dataset.b; state.hide.has(b) ? state.hide.delete(b) : state.hide.add(b); render(); });
  }
}

function renderSortOptions() {
  const keys = sortKeys();
  if (!keys.includes(state.sort)) state.sort = keys[0];
  $('sort').innerHTML = keys.map((k) => `<option value="${k}">${SORT_CN[k]}</option>`).join('');
  $('sort').value = state.sort;
}

function renderFooter() {
  $('prov').innerHTML = FWS.map((f) => {
    const v = data.frameworks[f];
    return `<b>${f}</b>：<code>${v.run_id}</code>${v.date ? `（${v.date}）` : ''}`;
  }).join('；') +
    `。两份都是 8 卡 B64、16K/512，模型 <code>qs-official-1017</code>。` +
    `构建脚本 <code>${DATASETS[state.ds].build}</code>。`;
  $('limits').innerHTML = data.limits.map((l) => `<li>${l}</li>`).join('');
}

function render() {
  renderSortOptions(); renderFilters(); renderDatasetNote(); renderOverview();
  renderStrips(); renderBuckets(); renderTables(); renderFooter(); writeURL();
}

$('ds').addEventListener('change', async (e) => { state.ds = e.target.value; await loadDataset(state.ds); render(); });
$('fw').addEventListener('change', (e) => { state.fw = e.target.value; render(); });
$('sort').addEventListener('change', (e) => { state.sort = e.target.value; render(); });

readURL();
await loadDataset(state.ds);
$('ds').value = state.ds; $('fw').value = state.fw;
render();
