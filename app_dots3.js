import {
  CFG, TILE, SMS, FULL_LAYERS, SWA_LAYERS,
  attnCfg, qkvAOut, projections, tiling, cost, attnScope, attnCost, kvPerToken, fmt, si,
} from './model_dots3.js';
import * as V41 from './model.js';

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
const ctxLabel = (c) => (c >= 1024 ? `${c / 1024}K` : String(c));

/* 状态写进 URL：?m=896&tp=1&ctx=14&layer=full&sel=QKV_A */
const state = { M: 896, tp: 1, ctxExp: 14, layer: 'full', sel: 'QKV_A' };
const VALID_SEL = ['QKV_A', 'Q_B', 'IDX_Q', 'W_KC', 'W_VC', 'O'];

function readURL() {
  const q = new URLSearchParams(location.search);
  const num = (k, lo, hi, dflt) => {
    const v = Number(q.get(k));
    return Number.isFinite(v) && v >= lo && v <= hi ? v : dflt;
  };
  state.M = Math.round(num('m', 64, 2432, 896) / 64) * 64;
  state.tp = [1, 2, 4, 8].includes(num('tp', 1, 8, 1)) ? num('tp', 1, 8, 1) : 1;
  state.ctxExp = Math.round(num('ctx', 10, 18, 14));
  if (['full', 'swa'].includes(q.get('layer'))) state.layer = q.get('layer');
  if (VALID_SEL.includes(q.get('sel'))) state.sel = q.get('sel');
  fixSel();
}
function fixSel() {
  // SWA 层没有 indexer，选中项落空时退回 QKV_A
  if (state.layer === 'swa' && state.sel === 'IDX_Q') state.sel = 'QKV_A';
}
function writeURL() {
  const q = new URLSearchParams({ m: state.M, tp: state.tp, ctx: state.ctxExp, layer: state.layer, sel: state.sel });
  history.replaceState(null, '', `${location.pathname}?${q}`);
}
const cur = () => projections(state.M, state.tp, state.layer);
const curSel = () => cur().find((q) => q.id === state.sel);

/* ------------------------------------------------------------------ A: 数据流 */
function renderFlow() {
  const a = attnCfg(state.layer);
  const ps = Object.fromEntries(cur().map((p) => [p.id, p]));
  const qa = qkvAOut(state.layer, state.tp);
  const s = attnScope(2 ** state.ctxExp, state.layer);
  const h = a.heads / state.tp;
  const W = 1420, H = 292;
  const g = [el('defs', {}, [
    el('marker', { id: 'ar', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: 'auto-start-reverse' },
      [el('path', { d: 'M0,0 L10,5 L0,10 z', fill: '#b3bcc4' })]),
  ])];

  const nodes = [
    { x: 0, y: 78, w: 100, h: 56, t: 'hidden', d: `[M, ${CFG.hidden_size}]`, k: 'act' },
    { x: 128, y: 78, w: 118, h: 56, t: 'QKV_A', d: `${CFG.hidden_size} → ${qa.total}`, k: 'w', id: 'QKV_A' },
    { x: 278, y: 30, w: 110, h: 50, t: 'q 潜向量', d: `[M, ${a.qLora}]`, k: 'act' },
    { x: 278, y: 126, w: 110, h: 50, t: 'KV 潜向量', d: `[M, ${qa.latent}]`, k: 'act' },
    { x: 420, y: 30, w: 110, h: 50, t: 'Q_B', d: `${a.qLora} → ${ps.Q_B.N}`, k: 'w', id: 'Q_B' },
    { x: 562, y: 30, w: 118, h: 50, t: 'query', d: `[M, ${h}, ${a.nope}+${a.rope}]`, k: 'act' },
    { x: 712, y: 30, w: 118, h: 50, t: 'W_KC', d: `${h} 头 · ${a.nope} → ${a.kvLora}`, k: 'w', id: 'W_KC' },
    { x: 866, y: 58, w: 176, h: 96, t: 'core attention', d: null, k: 'attn' },
    { x: 1066, y: 78, w: 112, h: 56, t: 'W_VC', d: `${h} 头 · ${a.kvLora} → ${a.v}`, k: 'w', id: 'W_VC' },
    { x: 1204, y: 78, w: 104, h: 56, t: 'O', d: `${ps.O.K} → ${CFG.hidden_size}`, k: 'w', id: 'O' },
    { x: 1334, y: 78, w: 86, h: 56, t: 'hidden', d: `[M, ${CFG.hidden_size}]`, k: 'act' },
  ];
  if (a.indexer) {
    nodes.push({ x: 420, y: 196, w: 150, h: 50, t: 'indexer wq_b', d: `${CFG.index_n_heads} 头 × ${CFG.index_head_dim}`, k: 'attn', id: 'IDX_Q' });
  } else {
    nodes.push({ x: 420, y: 196, w: 150, h: 50, t: '滑动窗口', d: `只看最近 ${a.window} 个 token`, k: 'attn' });
  }
  const FILL = { act: ['var(--act)', 'var(--act-edge)'], w: ['var(--w)', 'var(--w-edge)'], attn: ['var(--attn)', 'var(--attn-edge)'] };

  const edges = [
    [100, 106, 128, 106], [246, 100, 278, 60], [246, 112, 278, 148],
    [388, 55, 420, 55], [530, 55, 562, 55], [680, 55, 712, 55], [830, 55, 866, 86],
    [388, 151, 866, 126], [1042, 106, 1066, 106],
    [1178, 106, 1204, 106], [1308, 106, 1334, 106],
  ];
  if (a.indexer) edges.push([388, 60, 420, 215]);   // q 潜向量 → indexer wq_b
  for (const [x1, y1, x2, y2] of edges) g.push(el('path', { class: 'edge', d: `M${x1},${y1} L${x2},${y2}` }));
  // indexer / 窗口 -> core attention（虚线：只决定看哪些 token，不产生 value）
  g.push(el('path', { d: 'M570,221 L876,160', stroke: 'var(--attn-edge)', 'stroke-width': 1.3, 'stroke-dasharray': '3 3', fill: 'none', 'marker-end': 'url(#ar)' }));

  for (const n of nodes) {
    const [fill, stroke] = FILL[n.k];
    const cls = 'box' + (n.id === state.sel ? ' sel' : '');
    const kids = [rect(0, 0, n.w, n.h, fill, stroke)];
    if (n.t === 'core attention') {
      kids.push(txt(n.w / 2, 26, 'core attention', 'lbl', 'middle'));
      kids.push(txt(n.w / 2, 45, `${h} 头 · QK ${a.kvLora + a.rope} 维 / PV ${a.kvLora} 维`, 'dim', 'middle'));
      kids.push(txt(n.w / 2, 61, `每行看 ${fmt(s.selected)} 个 token`, 'dim', 'middle'));
      kids.push(txt(n.w / 2, 77, `（上下文 ${ctxLabel(s.context)}）`, 'dim', 'middle'));
    } else {
      kids.push(txt(n.w / 2, n.d ? n.h / 2 - 3 : n.h / 2 + 4, n.t, 'lbl', 'middle'));
      if (n.d) kids.push(txt(n.w / 2, n.h / 2 + 14, n.d, 'dim', 'middle'));
    }
    const grp = el('g', { class: cls, transform: `translate(${n.x},${n.y})` }, kids);
    if (n.id) grp.addEventListener('click', () => { state.sel = n.id; render(); });
    g.push(grp);
  }
  g.push(txt(278, 22, `${qa.total} = ${a.qLora} + ${a.kvLora} + ${qa.ropePad} + gate ${qa.gate}`, 'dim'));
  g.push(txt(0, 204, `KV 潜向量只有 1 份（${a.kvLora} + rope ${a.rope}），`, 'dim'));
  g.push(txt(0, 220, `不展开成 ${a.heads} 个头；每 token 每层`, 'dim'));
  g.push(txt(0, 236, `写进 KV cache 的就是这 ${qa.latent} 个元素`, 'dim'));
  g.push(txt(582, 226, a.indexer
    ? `对全部 ${fmt(s.scanned)} 个 token 打分，选 ${fmt(s.selected)} 个`
    : `窗口 ${a.window} = sliding_window_size − 1，不用 indexer`, 'dim'));

  $('flow').replaceChildren(svg(W, H, g));
}

/* ------------------------------------------- B: 选中步骤的矩阵乘法（按真实比例） */
function renderMatrix() {
  const p = curSel();
  const c = cost(p);
  const t = tiling(p);
  const PAD = 60, MAXW = 540, MAXH = 320, gap = 16;
  const s = Math.min(MAXW / (p.K + p.N), MAXH / (p.K + p.M));
  const kw = p.K * s, nw = p.N * s, mh = p.M * s;
  const x0 = PAD, y0 = 26;
  const xW = x0 + kw + gap, yA = y0 + kw + gap;
  const W = xW + nw + PAD, H = yA + mh + 40;

  const g = [];
  g.push(rect(xW, y0, nw, kw, 'var(--w)', 'var(--w-edge)'));
  if (p.perGroupN) {
    for (let i = 1; i < p.groups; i++) {
      const gx = xW + (nw * i) / p.groups;
      g.push(el('line', { x1: gx, y1: y0, x2: gx, y2: y0 + kw, stroke: 'var(--w-edge)', 'stroke-width': .6, opacity: .55 }));
    }
  }
  if (p.shard) g.push(rect(xW, y0, nw, kw, 'none', 'var(--warn)', { 'stroke-dasharray': '4 3', 'stroke-width': 1.5 }));
  g.push(rect(x0, yA, kw, mh, 'var(--act)', 'var(--act-edge)'));
  g.push(rect(xW, yA, nw, mh, '#e9f3ea', '#5a9e6f'));

  g.push(txt(xW + nw / 2, y0 - 9, `N = ${fmt(p.N)}` + (p.perGroupN ? `（${p.groups} 组 × ${p.perGroupN}）` : ''), 'dim', 'middle'));
  g.push(txt(xW - 7, y0 + kw / 2, `K = ${fmt(p.K)}`, 'dim', 'end'));
  if (kw >= 30) g.push(txt(x0 + kw / 2, yA - 7, `K = ${fmt(p.K)}`, 'dim', 'middle'));
  g.push(txt(x0 - 7, yA + mh / 2, `M = ${fmt(p.M)}`, 'dim', 'end'));
  g.push(txt(x0 + kw / 2, yA + mh + 17, '输入激活', 'dim', 'middle'));
  g.push(txt(xW + nw / 2, yA + mh + 17, `输出 [M, ${fmt(p.N)}] —— 切成 ${fmt(t.tiles)} 个 ${TILE}×${TILE} 方块`, 'dim', 'middle'));
  if (kw >= 22) g.push(txt(xW + nw / 2, y0 + kw / 2 + 4, p.dtype.toUpperCase() + ' 权重', 'dim', 'middle'));
  if (p.shard) g.push(txt(xW + nw + 6, y0 + kw / 2, `TP${state.tp} 切 ${p.shard}`, 'dim'));

  $('matrix').replaceChildren(svg(W, H, g));
  $('stepTitle').innerHTML = `<b>${p.label}</b> <span class="mono">${p.expr.replace(/\bM\b/g, fmt(p.M))}</span>`;
  $('stepWhy').innerHTML = p.why +
    (p.shard ? ` 当前 TP${state.tp}：全局 K=${fmt(p.globalK)}、N=${fmt(p.globalN)}，本卡 K=${fmt(p.K)}、N=${fmt(p.N)}。` : '');

  const rows = [
    ['乘加量 (FLOPs)', si(c.flops, 'FLOP'), '2·M·K·N'],
    ['权重字节', si(c.wBytes, 'B'), 'K·N×2B（BF16 checkpoint）'],
    ['输入激活字节', si(c.actIn, 'B'), p.perGroupN ? 'M·K×组数×2B' : 'M·K×2B'],
    ['输出激活字节', si(c.actOut, 'B'), 'M·N×2B'],
    ['算术强度', c.arithIntensity.toFixed(1) + ' FLOP/B', '越低越靠近带宽受限'],
  ];
  $('stepTable').innerHTML =
    '<tr><th>量</th><th>值</th><th style="text-align:left">怎么算的</th></tr>' +
    rows.map(([a, b, d]) => `<tr><td>${a}</td><td>${b}</td><td style="text-align:left;color:var(--grey)">${d}</td></tr>`).join('');
}

/* ------------------------------------------------------- C: 分块网格与 SM 占用 */
function renderTiles() {
  const p = curSel();
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
      : ` 轮数是向上取整的，所以 M 跨过 ${TILE} 的倍数时可能突然多一整轮。`) +
    (p.perGroupN
      ? ` W_KC / W_VC 每组的 N 只有 ${p.perGroupN}，按 ${TILE} 分块后每组 ${t.colsPerGroup} 列，` +
        `块数被头数撑起来、但每块很薄（K=${p.K}），真实 kernel 不会按这个 tile 跑，这里只给量级。`
      : '');
}

/* --------------------------------------------------- D: 这一层所有投影的总表 */
function renderAll() {
  const ps = cur();
  let totTiles = 0, totWaves = 0, totFlops = 0, totW = 0;
  const body = ps.map((p) => {
    const t = tiling(p), c = cost(p);
    totTiles += t.tiles; totWaves += t.waves; totFlops += c.flops; totW += c.wBytes;
    return `<tr class="${p.id === state.sel ? 'sel' : ''}" data-id="${p.id}" style="cursor:pointer">
      <td><span class="swatch" style="background:var(--w)"></span>${p.label}</td>
      <td>${fmt(p.K)}</td><td>${fmt(p.N)}${p.perGroupN ? ` <span style="color:var(--grey)">(${p.groups}×${p.perGroupN})</span>` : ''}</td>
      <td>${p.shard ? 'TP 切 ' + p.shard : '不切'}</td>
      <td>${t.cols}×${t.rows}</td><td>${fmt(t.tiles)}</td>
      <td>${t.waves}</td>
      <td class="${t.occupancy < 0.75 ? 'hl' : ''}">${(t.occupancy * 100).toFixed(0)}%</td>
      <td>${si(c.wBytes, 'B')}</td>
      <td>${si(c.flops, 'F')}</td></tr>`;
  }).join('');
  $('allTable').innerHTML =
    `<tr><th>矩阵乘法</th><th>K</th><th>N</th><th>TP</th><th>块排布</th><th>块数</th><th>轮数</th>
     <th>末轮占用</th><th>本卡权重</th><th>FLOPs</th></tr>${body}
     <tr class="total"><td>合计</td><td></td><td></td><td></td><td></td>
       <td>${fmt(totTiles)}</td><td>${totWaves}</td><td></td><td>${si(totW, 'B')}</td><td>${si(totFlops, 'F')}</td></tr>`;
  for (const tr of $('allTable').querySelectorAll('tr[data-id]')) {
    tr.addEventListener('click', () => { state.sel = tr.dataset.id; render(); });
  }

  const sum = (tp, layer) => projections(state.M, tp, layer).reduce((a, p) => a + tiling(p).waves, 0);
  const a = attnCfg(state.layer);
  $('allNote').innerHTML =
    `同样 M=${fmt(state.M)}，${state.layer === 'full' ? 'full' : 'SWA'} 层 <b>DP 合计 ${sum(1, state.layer)} 轮，TP4 合计 ${sum(4, state.layer)} 轮</b>` +
    `（另一类层：DP ${sum(1, state.layer === 'full' ? 'swa' : 'full')} 轮 / TP4 ${sum(4, state.layer === 'full' ? 'swa' : 'full')} 轮）。` +
    `TP 把 N 切小以后每张卡的方块少了，但四张卡要各跑一遍，而且 O 之后还要 all-reduce。` +
    `这张表只数方块，不代表时间 —— 每个方块里的活跟 K 成正比；` +
    `W_KC 的 K 只有 ${a.nope}，块最多但每块最浅。`;
}

/* ------------------------------------------------------------ E: 每行看多少 token */
function renderFunnel() {
  const ctx = 2 ** state.ctxExp;
  const a = attnCfg(state.layer);
  const s = attnScope(ctx, state.layer);
  const c = attnCost(state.M, ctx, state.tp, state.layer);
  const W = 1420, H = state.layer === 'full' ? 156 : 112, X = 360, BW = 910;
  const lg = (v) => Math.log2(Math.max(v, 1)) / Math.log2(CFG.max_position_embeddings);
  const bars = [['上下文全长', s.context, '#c9d6e0', `模型上限 ${fmt(CFG.max_position_embeddings)} = 384K`]];
  if (state.layer === 'full') {
    bars.push([`indexer 打分范围`, s.scanned, '#7fb0d4', `${CFG.index_n_heads} 头 × ${CFG.index_head_dim}，对每个 token 打分，没有封顶`]);
    bars.push([`indexer 选中：index_topk`, s.selected, 'var(--busy)', `单位是 token；上下文短于 ${fmt(CFG.index_topk)} 时全看`]);
  } else {
    bars.push([`滑动窗口`, s.selected, 'var(--busy)', `sliding_window_size − 1 = ${a.window}，不用 indexer`]);
  }
  const g = [];
  bars.forEach(([name, v, col, note], i) => {
    const y = 14 + i * 44, w = Math.max(lg(v) * BW, 2);
    g.push(txt(X - 12, y + 10, name, 'lbl', 'end'));
    g.push(txt(X - 12, y + 25, note, 'dim', 'end'));
    g.push(rect(X, y, w, 22, col, 'none', { rx: 3 }));
    g.push(txt(X + w + 9, y + 16, `${fmt(v)} token`, 'dim'));
  });
  $('funnel').replaceChildren(svg(W, H, g));

  const core = `按本卡 ${c.heads} 个头算，core attention 这一步是 <b>${si(c.total, 'FLOP')}</b>` +
    `（QK 在 ${a.kvLora + a.rope} 维潜空间、PV 在 ${a.kvLora} 维，吸收后不展开到每个头的 K/V）。`;
  $('funnelNote').innerHTML = state.layer === 'full'
    ? `条形是 log 宽度。当前每行实际参与打分的是 <b>${fmt(s.selected)}</b> 个 token，占全长 ` +
      `<b>${(s.keepRatio * 100).toFixed(2)}%</b>。${core}` +
      ` 但 indexer 要先对全部 <b>${fmt(s.scanned)}</b> 个 token 打分，这一步是 <b>${si(c.indexer, 'FLOP')}</b>` +
      `，是 core attention 的 <b>${(c.indexer / c.total).toFixed(2)}×</b>。` +
      `没有 V4.1 的候选块那一级，所以上下文越长 indexer 占比越高、不会和上下文长度脱钩` +
      `（V4.1 的候选集在 16,384 封顶）。`
    : `条形是 log 宽度。SWA 层每行只看最近 <b>${fmt(s.selected)}</b> 个 token，和上下文长度无关。${core}` +
      ` 这 33 层的 KV 只需保留最近 ${a.window} 个 token，所以长上下文下它们不占缓存，但每层每 token 的潜向量是 ${a.kvLora + a.rope} 个元素，比 full 层的 ${CFG.kv_lora_rank + CFG.qk_rope_head_dim} 宽。`;
}

/* --------------------------------------------------- F: 与 V4.1-Flash 并排 */
function renderCompare() {
  const M = state.M, tp = state.tp;
  const agg = (ps) => ps.reduce((o, p) => {
    const t = tiling(p), c = cost(p);
    o.tiles += t.tiles; o.waves += t.waves; o.flops += c.flops; o.w += c.wBytes; return o;
  }, { tiles: 0, waves: 0, flops: 0, w: 0 });
  const d = agg(cur());
  const dOther = agg(projections(M, tp, state.layer === 'full' ? 'swa' : 'full'));
  const v = agg(V41.projections(M, tp));
  const a = attnCfg(state.layer);
  const kv = kvPerToken();
  const v41Kv = V41.CFG.num_key_value_heads * V41.CFG.head_dim;   // 512 per layer（官方口径，不含 rope 拆分）
  const ctx = 2 ** state.ctxExp;
  const dScope = attnScope(ctx, 'full').selected, vScope = V41.sparseFunnel(ctx).selected;
  const row = (name, x, y, note = '') => `<tr><td>${name}</td><td>${x}</td><td>${y}</td><td style="text-align:left;color:var(--grey)">${note}</td></tr>`;
  const layerName = state.layer === 'full' ? 'full 层' : 'SWA 层';
  $('cmpTable').innerHTML =
    `<tr><th>项</th><th>dots3（${layerName}）</th><th>V4.1-Flash</th><th style="text-align:left">说明</th></tr>` +
    row('attention 头数 / 本卡', `${a.heads} / ${a.heads / tp}`, `${V41.CFG.num_attention_heads} / ${V41.CFG.num_attention_heads / tp}`, 'dots3 full 层头数是 V4.1 的 2 倍，但每头维度小') +
    row('每头 QK / V 维度', `${a.nope}+${a.rope} / ${a.v}`, `${V41.CFG.head_dim} / ${V41.CFG.head_dim}`, 'V4.1 没有 nope/rope 拆分字段，head_dim 512 含 rope 64') +
    row('每 token 每层 KV 潜向量', `${a.kvLora + a.rope}`, `${v41Kv}`, '元素数；dots3 = kv_lora_rank + rope，V4.1 = num_key_value_heads × head_dim') +
    row('每 token 全模型 KV', `${fmt(kv.totalElems)}（长期只留 ${fmt(kv.persistElems)}）`, `${fmt(v41Kv * V41.CFG.num_hidden_layers)}（CED 压缩后更少）`, 'dots3：13 full × 576 + 33 SWA × 1088；SWA 层只需最近 511 个 token') +
    row('每行实际看的 token（当前上下文）', `${fmt(dScope)}（full）/ ${fmt(attnScope(ctx, 'swa').selected)}（SWA）`, `${fmt(vScope)}（+SWA ${V41.CFG.sliding_window}）`, 'dots3 top-2048 单级；V4.1 候选 16,384 → top-512') +
    row('indexer 打分范围', `全部 ${fmt(ctx)}`, `≤ ${fmt(V41.CFG.candidate_topk_blocks * V41.CFG.candidate_block_size)}`, 'dots3 indexer 64 头，V4.1 32 头；dots3 没有候选块封顶') +
    row('这一层的矩阵乘法块数 / 轮数', `${fmt(d.tiles)} / ${d.waves}`, `${fmt(v.tiles)} / ${v.waves}`, `同 M=${fmt(M)}、TP${tp}；另一类层 ${fmt(dOther.tiles)} / ${dOther.waves}`) +
    row('这一层的投影 FLOPs', si(d.flops, 'F'), si(v.flops, 'F'), '只算本页画出的 GEMM/BMM，不含 core attention 和 indexer 打分') +
    row('这一层本卡 attention 权重', si(d.w, 'B'), si(v.w, 'B'), 'dots3 BF16；V4.1 FP8（含 scale）、O_A BF16') +
    row('attention 层数', `${kv.fullLayers} full + ${kv.swaLayers} SWA`, `${V41.CFG.num_hidden_layers}（20 enc + 20 dec，CED）`, '');
  $('cmpNote').innerHTML =
    `这一栏只比 config 推出的形状，不是测出来的时间。dots3 每层 attention 的投影算量比 V4.1 大，` +
    `主要来自 128 头 × (128+64) 的 Q_B 和吸收路径里两个按头分组的 BMM；` +
    `V4.1 把 O 拆成 O_A/O_B 低秩，dots3 的 O 是整块。` +
    `KV 方面，dots3 长期保留的只有 13 个 full 层，每 token ${fmt(kv.persistElems)} 个元素（BF16 ${(kv.persistElems * 2 / 1024).toFixed(1)} KB）；` +
    `V4.1 的 CED 压缩和 KV source layer 共享不在这张表里，所以两边的"全模型 KV"不能直接当作缓存占用来比。`;
}

/* -------------------------------------------------------------------- 控制 */
function renderControls() {
  const a = attnCfg(state.layer);
  const rb = Math.ceil(state.M / TILE);
  $('mOut').textContent = fmt(state.M);
  $('mHint').innerHTML = `${fmt(state.M)} ÷ ${TILE} = <b>${rb}</b> 块高` +
    (state.M % TILE === 0 ? '（正好整除）' : `（向上取整，浪费 ${rb * TILE - state.M} 行）`);
  $('tpHint').textContent = state.tp === 1
    ? `每张卡算全部 ${a.heads} 个头，不需要 all-reduce。`
    : `每张卡 ${a.heads / state.tp} 个头；Q_B、W_KC、W_VC 按头切，O 切 K，之后要 all-reduce。`;
  $('layerHint').textContent = state.layer === 'full'
    ? `层号 ${FULL_LAYERS.join(', ')}。q_lora ${a.qLora}、kv_lora ${a.kvLora}、nope ${a.nope}、rope ${a.rope}、v ${a.v}。`
    : `其余 ${SWA_LAYERS.length} 层。q_lora ${a.qLora}、kv_lora ${a.kvLora}、nope ${a.nope}、rope ${a.rope}、v ${a.v}，窗口 ${a.window}。`;
  const ctx = 2 ** state.ctxExp;
  $('ctxOut').textContent = ctxLabel(ctx);
  $('ctxHint').textContent = `${fmt(ctx)} token（模型上限 384K）`;
}

function render() {
  fixSel();
  renderControls(); renderFlow(); renderMatrix(); renderTiles(); renderAll(); renderFunnel(); renderCompare(); writeURL();
}

$('m').addEventListener('input', (e) => { state.M = +e.target.value; render(); });
$('tp').addEventListener('change', (e) => { state.tp = +e.target.value; render(); });
$('layer').addEventListener('change', (e) => { state.layer = e.target.value; render(); });
$('ctx').addEventListener('input', (e) => { state.ctxExp = +e.target.value; render(); });
for (const b of document.querySelectorAll('.presets button')) {
  b.addEventListener('click', () => { state.M = +b.dataset.m; $('m').value = state.M; render(); });
}

readURL();
$('m').value = state.M;
$('tp').value = state.tp;
$('layer').value = state.layer;
$('ctx').value = state.ctxExp;
render();
