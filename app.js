import {
  CFG, KV_LATENT, QKV_A_OUT, TILE, SMS,
  projections, tiling, cost, sparseFunnel, attnCost, fmt, si,
} from './model.js';

const NS = 'http://www.w3.org/2000/svg';
const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, kids = []) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) n.setAttribute(k, v);
  for (const c of [].concat(kids)) n.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  return n;
}
function svg(w, h, kids) {
  return el('svg', { viewBox: `0 0 ${w} ${h}`, preserveAspectRatio: 'xMidYMid meet' }, kids);
}
function txt(x, y, s, cls = 'lbl', anchor = 'start') {
  return el('text', { x, y, class: cls, 'text-anchor': anchor }, String(s));
}
function rect(x, y, w, h, fill, stroke, extra = {}) {
  return el('rect', { x, y, width: Math.max(w, 0), height: Math.max(h, 0), fill, stroke, 'stroke-width': 1.2, rx: 2, ...extra });
}

/* 状态可以写进 URL，方便把某一个具体配置的链接直接发给别人。
   例：?m=896&tp=4&sel=Q_B&ctx=14 */
const state = { M: 896, tp: 1, ctxExp: 14, sel: 'QKV_A' };
const VALID_SEL = ['QKV_A', 'Q_B', 'O_A', 'O_B'];

function readURL() {
  const q = new URLSearchParams(location.search);
  const num = (k, lo, hi, dflt) => {
    const v = Number(q.get(k));
    return Number.isFinite(v) && v >= lo && v <= hi ? v : dflt;
  };
  state.M = Math.round(num('m', 64, 2432, 896) / 64) * 64;
  state.tp = [1, 2, 4, 8].includes(num('tp', 1, 8, 1)) ? num('tp', 1, 8, 1) : 1;
  state.ctxExp = Math.round(num('ctx', 10, 20, 14));
  if (VALID_SEL.includes(q.get('sel'))) state.sel = q.get('sel');
}

function writeURL() {
  const q = new URLSearchParams({ m: state.M, tp: state.tp, ctx: state.ctxExp, sel: state.sel });
  history.replaceState(null, '', `${location.pathname}?${q}`);
}

/* ------------------------------------------------------------------ A: 数据流 */
function renderFlow() {
  const ps = Object.fromEntries(projections(state.M, state.tp).map((p) => [p.id, p]));
  const f = sparseFunnel(2 ** state.ctxExp);
  const W = 1420, H = 282;
  const g = [el('defs', {}, [
    el('marker', { id: 'ar', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' },
      [el('path', { d: 'M0,0 L10,5 L0,10 z', fill: '#b3bcc4' })]),
  ])];

  const heads = CFG.num_attention_heads / state.tp;
  const nodes = [
    { x: 0, y: 78, w: 112, h: 56, t: 'hidden', d: `[M, ${CFG.hidden_size}]`, k: 'act' },
    { x: 148, y: 78, w: 126, h: 56, t: 'QKV_A', d: `${CFG.hidden_size} → ${QKV_A_OUT}`, k: 'w', id: 'QKV_A' },
    { x: 312, y: 30, w: 118, h: 50, t: 'q 潜向量', d: `[M, ${CFG.q_lora_rank}]`, k: 'act' },
    { x: 312, y: 136, w: 118, h: 50, t: 'KV 潜向量', d: `[M, ${KV_LATENT}]`, k: 'act' },
    { x: 468, y: 30, w: 118, h: 50, t: 'Q_B', d: `${CFG.q_lora_rank} → ${ps.Q_B.N}`, k: 'w', id: 'Q_B' },
    { x: 624, y: 30, w: 118, h: 50, t: 'query', d: `[M, ${heads}, ${CFG.head_dim}]`, k: 'act' },
    { x: 468, y: 190, w: 150, h: 50, t: 'indexer', d: `${CFG.index_n_heads} 头 × ${CFG.index_head_dim}`, k: 'attn' },
    { x: 790, y: 58, w: 150, h: 96, t: 'core attention', d: null, k: 'attn' },
    { x: 990, y: 78, w: 126, h: 56, t: 'O_A', d: `${ps.O_A.groups} 组 · 4096 → ${CFG.o_lora_rank}`, k: 'w', id: 'O_A' },
    { x: 1154, y: 78, w: 126, h: 56, t: 'O_B', d: `${ps.O_B.K} → ${CFG.hidden_size}`, k: 'w', id: 'O_B' },
    { x: 1318, y: 78, w: 102, h: 56, t: 'hidden', d: `[M, ${CFG.hidden_size}]`, k: 'act' },
  ];
  const FILL = { act: ['var(--act)', 'var(--act-edge)'], w: ['var(--w)', 'var(--w-edge)'], attn: ['var(--attn)', 'var(--attn-edge)'] };

  const edges = [
    [112, 106, 148, 106], [274, 100, 312, 60], [274, 112, 312, 156],
    [430, 55, 468, 55], [586, 55, 624, 55], [742, 55, 790, 86],
    [430, 161, 790, 126], [940, 106, 990, 106],
    [1116, 106, 1154, 106], [1280, 106, 1318, 106],
  ];
  for (const [x1, y1, x2, y2] of edges) g.push(el('path', { class: 'edge', d: `M${x1},${y1} L${x2},${y2}` }));
  // indexer -> core attention（虚线：它只决定看哪些 token，不产生 value）
  g.push(el('path', { d: 'M543,190 L543,158', stroke: 'var(--attn-edge)', 'stroke-width': 1.3, 'stroke-dasharray': '3 3', fill: 'none', 'marker-end': 'url(#ar)' }));

  for (const n of nodes) {
    const [fill, stroke] = FILL[n.k];
    const cls = 'box' + (n.id === state.sel ? ' sel' : '');
    const kids = [rect(0, 0, n.w, n.h, fill, stroke)];
    if (n.t === 'core attention') {
      kids.push(txt(n.w / 2, 26, 'core attention', 'lbl', 'middle'));
      kids.push(txt(n.w / 2, 45, `${heads} 头 × ${CFG.head_dim}`, 'dim', 'middle'));
      kids.push(txt(n.w / 2, 61, `每行看 ${fmt(f.selected)} 个 token`, 'dim', 'middle'));
      kids.push(txt(n.w / 2, 77, `（上下文 ${f.context >= 1024 ? f.context / 1024 + 'K' : f.context}）`, 'dim', 'middle'));
    } else {
      kids.push(txt(n.w / 2, n.d ? n.h / 2 - 3 : n.h / 2 + 4, n.t, 'lbl', 'middle'));
      if (n.d) kids.push(txt(n.w / 2, n.h / 2 + 14, n.d, 'dim', 'middle'));
    }
    const grp = el('g', { class: cls, transform: `translate(${n.x},${n.y})` }, kids);
    if (n.id) grp.addEventListener('click', () => { state.sel = n.id; render(); });
    g.push(grp);
  }
  g.push(txt(312, 22, `${QKV_A_OUT} = ${CFG.q_lora_rank} + ${KV_LATENT}`, 'dim'));
  g.push(txt(634, 218, `从 ${fmt(f.candidates)} 个候选里选 ${fmt(f.selected)} 个 token`, 'dim'));
  g.push(txt(312, 204, `KV 只有 ${CFG.num_key_value_heads} 个头，不展开成 ${CFG.num_attention_heads} 份`, 'dim'));

  $('flow').replaceChildren(svg(W, H, g));
}

/* ------------------------------------------- B: 选中步骤的矩阵乘法（按真实比例） */
function renderMatrix() {
  const p = projections(state.M, state.tp).find((q) => q.id === state.sel);
  const c = cost(p);
  const t = tiling(p);
  const PAD = 60, MAXW = 540, MAXH = 320, gap = 16;
  const s = Math.min(MAXW / (p.K + p.N), MAXH / (p.K + p.M));
  const kw = p.K * s, nw = p.N * s, mh = p.M * s;
  const x0 = PAD, y0 = 26;
  const xW = x0 + kw + gap, yA = y0 + kw + gap;
  const W = xW + nw + PAD, H = yA + mh + 40;

  const g = [];
  // W: [K, N]
  g.push(rect(xW, y0, nw, kw, 'var(--w)', 'var(--w-edge)'));
  if (p.perGroupN) {
    for (let i = 1; i < p.groups; i++) {
      const gx = xW + (nw * i) / p.groups;
      g.push(el('line', { x1: gx, y1: y0, x2: gx, y2: y0 + kw, stroke: 'var(--w-edge)', 'stroke-width': .6, opacity: .55 }));
    }
  }
  if (p.shard === 'N') g.push(rect(xW, y0, nw, kw, 'none', 'var(--warn)', { 'stroke-dasharray': '4 3', 'stroke-width': 1.5 }));
  if (p.shard === 'K') g.push(rect(xW, y0, nw, kw, 'none', 'var(--warn)', { 'stroke-dasharray': '4 3', 'stroke-width': 1.5 }));
  // A: [M, K]
  g.push(rect(x0, yA, kw, mh, 'var(--act)', 'var(--act-edge)'));
  // Y: [M, N]
  g.push(rect(xW, yA, nw, mh, '#e9f3ea', '#5a9e6f'));

  g.push(txt(xW + nw / 2, y0 - 9, `N = ${fmt(p.N)}`, 'dim', 'middle'));
  g.push(txt(xW - 7, y0 + kw / 2, `K = ${fmt(p.K)}`, 'dim', 'end'));
  // K 很小时两个 K 标签会贴在一起，只留权重那一侧的
  if (kw >= 30) g.push(txt(x0 + kw / 2, yA - 7, `K = ${fmt(p.K)}`, 'dim', 'middle'));
  g.push(txt(x0 - 7, yA + mh / 2, `M = ${fmt(p.M)}`, 'dim', 'end'));
  g.push(txt(x0 + kw / 2, yA + mh + 17, '输入激活', 'dim', 'middle'));
  g.push(txt(xW + nw / 2, yA + mh + 17, `输出 [M, ${fmt(p.N)}] —— 切成 ${fmt(t.tiles)} 个 ${TILE}×${TILE} 方块`, 'dim', 'middle'));
  g.push(txt(xW + nw / 2, y0 + kw / 2 + 4, p.dtype.toUpperCase() + ' 权重', 'dim', 'middle'));
  if (p.shard) {
    g.push(txt(xW + nw + 6, y0 + kw / 2, `TP${state.tp} 切 ${p.shard}`, 'dim'));
  }

  $('matrix').replaceChildren(svg(W, H, g));
  // \b 两侧都要边界，所以 "BMM" 里的 M 不会被替换，只有 "[M, ...]" 里的会
  $('stepTitle').innerHTML = `<b>${p.label}</b> <span class="mono">${p.expr.replace(/\bM\b/g, fmt(p.M))}</span>`;
  $('stepWhy').innerHTML = p.why +
    (p.shard ? ` 当前 TP${state.tp}：全局 K=${fmt(p.globalK)}、N=${fmt(p.globalN)}，本卡 K=${fmt(p.K)}、N=${fmt(p.N)}。` : '');

  const rows = [
    ['乘加量 (FLOPs)', si(c.flops, 'FLOP'), '2·M·K·N'],
    ['权重字节', si(c.wBytes, 'B'), p.dtype === 'fp8' ? 'K·N×1B ＋ 每 32 个元素 1 个 scale' : 'K·N×2B'],
    ['输入激活字节', si(c.actIn, 'B'), 'M·K'],
    ['输出激活字节', si(c.actOut, 'B'), 'M·N×2B'],
    ['算术强度', c.arithIntensity.toFixed(1) + ' FLOP/B', '越低越靠近带宽受限'],
  ];
  $('stepTable').innerHTML =
    '<tr><th>量</th><th>值</th><th style="text-align:left">怎么算的</th></tr>' +
    rows.map(([a, b, d]) => `<tr><td>${a}</td><td>${b}</td><td style="text-align:left;color:var(--grey)">${d}</td></tr>`).join('');
}

/* ------------------------------------------------------- C: 瓦片网格与 SM 占用 */
function renderTiles() {
  const p = projections(state.M, state.tp).find((q) => q.id === state.sel);
  const t = tiling(p);
  const AVW = 600;
  const cell = Math.max(1.6, Math.min(13, (AVW - 10) / t.cols));
  const gapPx = cell > 4 ? cell * 0.12 : 0;
  const W = AVW, H = t.rows * cell + 34;
  const g = [];
  for (let r = 0; r < t.rows; r++) {
    for (let c = 0; c < t.cols; c++) {
      const idx = r * t.cols + c;
      const last = idx >= (t.waves - 1) * SMS;
      g.push(rect(c * cell, 22 + r * cell, cell - gapPx, cell - gapPx,
        last && t.waves > 1 ? '#8fb9dc' : 'var(--busy)', 'none', { rx: cell > 5 ? 1.5 : 0 }));
    }
  }
  g.push(txt(0, 13, `${t.cols} 块宽 × ${t.rows} 块高 = ${fmt(t.tiles)} 块` +
    (t.waves > 1 ? `（浅色 = 最后一轮的 ${t.lastWave} 块）` : ''), 'dim'));
  $('tiles').replaceChildren(svg(W, H, g));
  $('tileHead').innerHTML =
    `<b>${p.label}</b> 的输出：N=${fmt(p.N)} ÷ ${TILE} = <b>${t.cols}</b> 块宽` +
    (p.perGroupN ? `（${p.groups} 组 × ${t.colsPerGroup}）` : '') +
    `，M=${fmt(p.M)} ÷ ${TILE} = <b>${t.rows}</b> 块高`;

  // SM strip：轮数少时把格子画高一点，否则一轮只有一条细线看不出东西
  const SW = 600, cw = (SW - 8) / SMS;
  const rowH = t.waves <= 3 ? 22 : t.waves <= 10 ? 12 : 7;
  const showWaves = Math.min(t.waves, 40);
  const sg = [];
  for (let w = 0; w < showWaves; w++) {
    const inThis = w === t.waves - 1 ? t.lastWave : SMS;
    for (let i = 0; i < SMS; i++) {
      sg.push(rect(i * cw, 20 + w * rowH, cw - .8, rowH - 1.6,
        i < inThis ? (w === t.waves - 1 ? '#8fb9dc' : 'var(--busy)') : 'var(--idle)', 'none', { rx: 1 }));
    }
  }
  if (t.waves > showWaves) sg.push(txt(0, 20 + showWaves * rowH + 12, `…… 还有 ${t.waves - showWaves} 轮`, 'dim'));
  sg.push(txt(0, 12, `${SMS} 个 SM × ${t.waves} 轮` + (t.waves > 1 ? '（浅色 = 最后一轮）' : ''), 'dim'));
  $('sms').replaceChildren(svg(SW, 20 + showWaves * rowH + (t.waves > showWaves ? 20 : 8), sg));
  $('smHead').innerHTML = `需要 <b>${t.waves}</b> 轮；最后一轮只用上 <b>${t.lastWave}</b> / ${SMS} 个 SM` +
    `（<b class="${t.occupancy < 0.75 ? 'hl' : ''}">${(t.occupancy * 100).toFixed(1)}%</b>）`;

  const waste = (SMS - t.lastWave) / (t.waves * SMS);
  $('tileNote').innerHTML =
    `整体看，${t.waves} 轮里有 <b>${fmt(SMS * t.waves - t.tiles)}</b> 个 SM-轮次是空的，占 ` +
    `<b>${(waste * 100).toFixed(1)}%</b>。` +
    (t.waves === 1
      ? ` 只有一轮时，M 再往上加也不会多排一轮 —— 多出来的方块直接填进空转的核里，` +
        `一直要到 M 超过 <b>${fmt(TILE * Math.floor(SMS / t.cols))}</b> 行才排到第二轮。`
      : ` 轮数是向上取整的，所以 M 跨过 ${TILE} 的倍数时可能突然多一整轮。`);
}

/* --------------------------------------------------- D: 四个投影放在一起的总表 */
function renderAll() {
  const ps = projections(state.M, state.tp);
  let totTiles = 0, totWaves = 0, totFlops = 0;
  const body = ps.map((p) => {
    const t = tiling(p), c = cost(p);
    totTiles += t.tiles; totWaves += t.waves; totFlops += c.flops;
    return `<tr class="${p.id === state.sel ? 'sel' : ''}" data-id="${p.id}" style="cursor:pointer">
      <td><span class="swatch" style="background:var(--w)"></span>${p.label}</td>
      <td>${fmt(p.K)}</td><td>${fmt(p.N)}</td>
      <td>${p.shard ? 'TP 切 ' + p.shard : '不切'}</td>
      <td>${t.cols}×${t.rows}</td><td>${fmt(t.tiles)}</td>
      <td>${t.waves}</td>
      <td class="${t.occupancy < 0.75 ? 'hl' : ''}">${(t.occupancy * 100).toFixed(0)}%</td>
      <td>${si(c.flops, 'F')}</td></tr>`;
  }).join('');
  $('allTable').innerHTML =
    `<tr><th>投影</th><th>K</th><th>N</th><th>TP</th><th>块排布</th><th>块数</th><th>轮数</th>
     <th>末轮占用</th><th>FLOPs</th></tr>${body}
     <tr class="total"><td>合计</td><td></td><td></td><td></td><td></td>
       <td>${fmt(totTiles)}</td><td>${totWaves}</td><td></td><td>${si(totFlops, 'F')}</td></tr>`;
  for (const tr of $('allTable').querySelectorAll('tr[data-id]')) {
    tr.addEventListener('click', () => { state.sel = tr.dataset.id; render(); });
  }

  const dp = projections(state.M, 1).reduce((a, p) => a + tiling(p).waves, 0);
  const tp4 = projections(state.M, 4).reduce((a, p) => a + tiling(p).waves, 0);
  $('allNote').innerHTML =
    `同样 M=${fmt(state.M)}，<b>DP 合计 ${dp} 轮，TP4 合计 ${tp4} 轮</b>。` +
    `TP 把 N 切小以后每张卡的方块少了，但四张卡要各跑一遍，而且 O_B 之后还要 all-reduce。` +
    `这张表只数方块，不代表时间 —— 每个方块里的活跟 K 成正比，` +
    `Q_B 的 K 只有 ${CFG.q_lora_rank}，块多但每块很浅。`;
}

/* ------------------------------------------------------------ E: 稀疏两级漏斗 */
function renderFunnel() {
  const f = sparseFunnel(2 ** state.ctxExp);
  const a = attnCost(state.M, f.selected, state.tp);
  const W = 1420, H = 156, X = 330, BW = 940;
  const lg = (v) => Math.log2(Math.max(v, 1)) / Math.log2(CFG.max_position_embeddings);
  const bars = [
    ['上下文全长', f.context, '#c9d6e0', `模型上限 ${CFG.max_position_embeddings / 1024}K`],
    [`第一级候选：${CFG.candidate_topk_blocks} 块 × ${CFG.candidate_block_size}`, f.candidates, '#7fb0d4',
      `由第 ${CFG.candidate_source_layer_id} 层产生，封顶 ${fmt(CFG.candidate_topk_blocks * CFG.candidate_block_size)}`],
    ['第二级 indexer 选中：index_topk', f.selected, 'var(--busy)',
      `${CFG.index_n_heads} 头 × ${CFG.index_head_dim}；单位是 token，不是 block`],
  ];
  const g = [];
  bars.forEach(([name, v, col, note], i) => {
    const y = 14 + i * 44, w = Math.max(lg(v) * BW, 2);
    g.push(txt(X - 12, y + 10, name, 'lbl', 'end'));
    g.push(txt(X - 12, y + 25, note, 'dim', 'end'));
    g.push(rect(X, y, w, 22, col, 'none', { rx: 3 }));
    g.push(txt(X + w + 9, y + 16, `${fmt(v)} token`, 'dim'));
  });
  $('funnel').replaceChildren(svg(W, H, g));
  $('funnelNote').innerHTML =
    `条形是 log 宽度，否则 ${fmt(f.selected)} 在 ${fmt(f.context)} 面前看不见。` +
    `当前每行实际参与打分的是 <b>${fmt(f.selected)}</b> 个 token，占全长 ` +
    `<b>${(f.keepRatio * 100).toFixed(3)}%</b>。` +
    `按本卡 ${a.heads} 个头算，core attention 这一步是 <b>${si(a.total, 'FLOP')}</b>` +
    `（QK 和 PV 各一半；MLA 里 K 和 V 共用同一份 ${CFG.head_dim} 维潜向量，所以两边形状一样）。` +
    `上下文超过 ${fmt(CFG.candidate_topk_blocks * CFG.candidate_block_size)} 之后，` +
    `候选集不再增长，第二级的工作量就和上下文长度脱钩了。`;
}

/* -------------------------------------------------------------------- 控制 */
function renderControls() {
  const rb = Math.ceil(state.M / TILE);
  $('mOut').textContent = fmt(state.M);
  $('mHint').innerHTML = `${fmt(state.M)} ÷ ${TILE} = <b>${rb}</b> 块高` +
    (state.M % TILE === 0 ? '（正好整除）' : `（向上取整，浪费 ${rb * TILE - state.M} 行）`);
  $('tpHint').textContent = state.tp === 1
    ? '每张卡算全部 64 个头，不需要 all-reduce。'
    : `每张卡 ${CFG.num_attention_heads / state.tp} 个头、${CFG.o_groups / state.tp} 个 O_A 分组；O_B 之后要 all-reduce。`;
  const ctx = 2 ** state.ctxExp;
  $('ctxOut').textContent = ctx >= 1024 ? `${ctx / 1024}K` : ctx;
  $('ctxHint').textContent = `${fmt(ctx)} token`;
}

function render() {
  renderControls(); renderFlow(); renderMatrix(); renderTiles(); renderAll(); renderFunnel(); writeURL();
}

$('m').addEventListener('input', (e) => { state.M = +e.target.value; render(); });
$('tp').addEventListener('change', (e) => { state.tp = +e.target.value; render(); });
$('ctx').addEventListener('input', (e) => { state.ctxExp = +e.target.value; render(); });
for (const b of document.querySelectorAll('.presets button')) {
  b.addEventListener('click', () => { state.M = +b.dataset.m; $('m').value = state.M; render(); });
}

readURL();
$('m').value = state.M;
$('tp').value = state.tp;
$('ctx').value = state.ctxExp;
render();
