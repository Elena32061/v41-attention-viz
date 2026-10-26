const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const fmt = (n, d = 0) => n.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
const COLORS = {
  moe_grouped_gemm: '#0f6ab4', dense_projection: '#c8812a', attention: '#2a9d8f', mhc: '#8e5ea2',
  quantize_norm: '#d4a017', routing_dispatch: '#e76f51', collective: '#d1495b', eager_glue: '#9aa3ab',
  sampling: '#5a9e6f', swiglu: '#6c8ebf', engram: '#b5651d', topk_unresolved: '#c06c84', other: '#777',
};
const BUCKET_CN = {
  moe_grouped_gemm: 'MoE 专家 GEMM', dense_projection: '稠密投影', attention: 'attention', mhc: 'mHC',
  quantize_norm: '量化 / norm', routing_dispatch: '路由 / dispatch', collective: '集合通信', eager_glue: '胶水小 kernel',
  sampling: '采样', swiglu: 'SwiGLU', engram: 'Engram', topk_unresolved: 'top-k', other: '其他',
};
const state = { fw: 'both', sort: 'ms', hide: new Set() };
const data = await (await fetch('./kernels_b64_8card.json')).json();
const FWS = ['SGLang', 'DeepInfer'];
const r0 = (fw) => data.frameworks[fw].ranks['node_0/rank0'];
const r4 = (fw) => data.frameworks[fw].ranks['node_1/rank4'];
const buckets = [...new Set(FWS.flatMap((f) => r0(f).table.map((x) => x.bucket)))].sort(
  (a, b) => (COLORS[a] ? 0 : 1) - (COLORS[b] ? 0 : 1) || a.localeCompare(b));

function readURL() {
  const q = new URLSearchParams(location.search);
  if (['both', ...FWS].includes(q.get('fw'))) state.fw = q.get('fw');
  if (['ms', 'count', 'us', 'bucket'].includes(q.get('sort'))) state.sort = q.get('sort');
  if (q.get('hide')) state.hide = new Set(q.get('hide').split(',').filter((b) => buckets.includes(b)));
}
function writeURL() {
  const q = new URLSearchParams({ fw: state.fw, sort: state.sort });
  if (state.hide.size) q.set('hide', [...state.hide].join(','));
  history.replaceState(null, '', `${location.pathname}?${q}`);
}
const shown = () => (state.fw === 'both' ? FWS : [state.fw]);

function renderOverview() {
  const rows = [
    ['每步 kernel 数（中位 / 最小–最大）', (r) => `${r.kernels_median}（${r.kernels_min}–${r.kernels_max}）`],
    ['其中在 CUDA graph 里 / 外', (r) => `${r.modal_in_graph} / ${r.modal_outside_graph}`],
    ['100 步里 kernel 集合完全相同的步数', (r) => `${r.identical_multiset_steps}`],
    ['100 步里 kernel 顺序也完全相同的步数', (r) => `${r.identical_sequence_steps}`],
    ['不同 kernel 名数（rank 0）', (r, f) => `${r0(f).distinct_names}`],
    ['每步累计 kernel 时长（rank 0）', (r, f) => `${fmt(r0(f).total_ms_per_step_cumulative, 2)} ms`],
    ['profiler 开着时的步长', (r) => `${fmt(r.step_ms_mean, 2)} ms`],
    ['集合有差异的步里变化的 kernel', (r) => Object.entries(r.varying_kernels).map(([k, v]) => `${k.split('::').pop()}（${v} 步）`).join('<br>') || '无'],
  ];
  const head = `<tr><th></th>${FWS.flatMap((f) => [`<th>${f}<br><span style="font-weight:400">rank 0</span></th>`, `<th>${f}<br><span style="font-weight:400">rank 4</span></th>`]).join('')}</tr>`;
  $('overview').innerHTML = head + rows.map(([n, fn]) =>
    `<tr><td>${n}</td>${FWS.flatMap((f) => [`<td>${fn(r0(f), f)}</td>`, `<td>${fn(r4(f), f)}</td>`]).join('')}</tr>`).join('');
  const sg = r0('SGLang'), di = r0('DeepInfer');
  $('overviewNote').innerHTML =
    `<b>每步跑的 kernel 集合都一样。</b>SGLang 100 步集合相同，但顺序每步都不同 —— 它多流并发，不同 stream 上 kernel 的启动先后每步会变；` +
    `DeepInfer 99 步顺序一模一样，剩下 1 步多了几个 reduce / elementwise 和一次 AllGather。` +
    `SGLang 一步 ${sg.kernels_median} 个 kernel，DeepInfer ${di.kernels_median} 个，差 ${(sg.kernels_median / di.kernels_median).toFixed(1)} 倍，` +
    `差在 SGLang 把 mHC、norm、量化、胶水拆成很多小 kernel。两个节点上的 rank 0 和 rank 4 数字一致。`;
}

function renderStrips() {
  $('legend').innerHTML = buckets.map((b) => `<span style="--c:${COLORS[b] || COLORS.other}">${BUCKET_CN[b] || b}</span>`).join('');
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
  const html = [];
  for (const f of shown()) {
    const t = r0(f).table;
    const agg = {};
    for (const x of t) { const a = agg[x.bucket] ??= { ms: 0, n: 0, kinds: 0 }; a.ms += x.ms_per_step; a.n += x.count_per_step; a.kinds += 1; }
    const tot = Object.values(agg).reduce((s, a) => s + a.ms, 0);
    const rows = Object.entries(agg).sort((a, b) => b[1].ms - a[1].ms).map(([b, a]) =>
      `<tr class="${state.hide.has(b) ? '' : ''}"><td><span class="swatch" style="background:${COLORS[b] || COLORS.other}"></span>${BUCKET_CN[b] || b}</td>` +
      `<td>${fmt(a.n, a.n % 1 ? 2 : 0)}</td><td>${a.kinds}</td><td>${fmt(a.ms, 3)}</td><td>${fmt((a.ms / tot) * 100, 1)}%</td></tr>`).join('');
    const sec = document.createElement('div');
    sec.innerHTML = `<div class="fw-title">${f}</div><table><tr><th>功能</th><th>次/步</th><th>种类</th><th>累计 ms/步</th><th>占累计</th></tr>${rows}` +
      `<tr class="total"><td>合计</td><td>${fmt(t.reduce((s, x) => s + x.count_per_step, 0), 0)}</td><td>${t.length}</td><td>${fmt(tot, 3)}</td><td></td></tr></table>`;
    html.push(sec);
  }
  $('buckets').replaceChildren(...html);
}

function renderTables() {
  const key = { ms: (x) => -x.ms_per_step, count: (x) => -x.count_per_step, us: (x) => -x.us_mean, bucket: (x) => 0 }[state.sort];
  const html = [];
  for (const f of shown()) {
    let t = r0(f).table.filter((x) => !state.hide.has(x.bucket));
    t = [...t].sort((a, b) => (state.sort === 'bucket' ? (a.bucket.localeCompare(b.bucket) || b.ms_per_step - a.ms_per_step) : key(a) - key(b)));
    const rows = t.map((x) =>
      `<tr><td class="name" title="${x.short}"><span class="swatch" style="background:${COLORS[x.bucket] || COLORS.other}"></span>${x.short}</td>` +
      `<td>${fmt(x.count_per_step, x.count_per_step % 1 ? 2 : 0)}</td><td>${fmt(x.ms_per_step, 4)}</td><td>${fmt(x.us_mean, 1)}</td>` +
      `<td>${x.in_graph_frac >= 0.999 ? '内' : x.in_graph_frac <= 0.001 ? '<b class="hl">外</b>' : fmt(x.in_graph_frac * 100, 0) + '%'}</td></tr>`).join('');
    const sec = document.createElement('div');
    sec.innerHTML = `<div class="fw-title">${f} <span class="mono">${t.length} 种</span></div><div class="scroll"><table>` +
      `<tr><th>kernel</th><th>次/步</th><th>累计 ms/步</th><th>单次 μs</th><th>graph</th></tr>${rows}</table></div>`;
    html.push(sec);
  }
  $('tables').replaceChildren(...html);
  $('sortHint').textContent = { ms: '累计 = 次数 × 单次时长，不是墙钟。', count: '一步里被启动的次数。', us: '单个 kernel 的平均时长。', bucket: '同组内再按累计时间。' }[state.sort];
}

function renderFilters() {
  $('filters').innerHTML = buckets.map((b) =>
    `<button data-b="${b}" class="${state.hide.has(b) ? '' : 'on'}" style="border-left:4px solid ${COLORS[b] || COLORS.other}">${BUCKET_CN[b] || b}</button>`).join('');
  for (const btn of $('filters').querySelectorAll('button')) {
    btn.addEventListener('click', () => { const b = btn.dataset.b; state.hide.has(b) ? state.hide.delete(b) : state.hide.add(b); render(); });
  }
}

function renderFooter() {
  $('prov').innerHTML = FWS.map((f) => `<b>${f}</b>：<code>${data.frameworks[f].run_id}</code>`).join('；') +
    `。两份都是 8 卡 B64、16K/512，模型 <code>qs-official-1017</code>。统计目录 <code>outputs/sg_di_8card_kernel_compare_20261025/</code>。`;
  $('limits').innerHTML = data.limits.map((l) => `<li>${l}</li>`).join('');
}

function render() { renderFilters(); renderOverview(); renderStrips(); renderBuckets(); renderTables(); writeURL(); }
$('fw').addEventListener('change', (e) => { state.fw = e.target.value; render(); });
$('sort').addEventListener('change', (e) => { state.sort = e.target.value; render(); });
readURL(); $('fw').value = state.fw; $('sort').value = state.sort; renderFooter(); render();
