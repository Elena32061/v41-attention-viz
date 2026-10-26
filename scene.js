/* V4.1-Flash attention 的 3D 视图。
 *
 * 比例约定：1 个场景单位 = 128 个元素，正好是一个输出块的边长。于是
 *   宽(X) = N/128 = 块的列数    高(Y) = M/128 = 块的行数    深(Z) = K/128
 * 输出面天然就是块网格，K 变成真实的深度 —— 每个块要做多少乘加一眼可见。
 *
 * 几何全部来自 model.js（config 维度 + 形状算术），没有实测成分。
 * 动画只是固定速度地把一轮一轮刷过去，不代表任何真实 kernel 耗时。
 */
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import { CFG, TILE, SMS, projections, tiling, cost, fmt, si } from './model.js';

const $ = (id) => document.getElementById(id);
const GAP = 24;          // 两个体积之间的间距（场景单位）
const GROUP_GAP = 1.2;   // O_A 的 BMM 分组之间留一道缝
const SM_COLS = 19, SM_PITCH = 1.15;              // 152 = 19 × 8
const SM_W = SM_COLS * SM_PITCH;                  // SM 阵的宽
const SM_H = (SMS / SM_COLS) * SM_PITCH;          // SM 阵的高
const SM_TOP = -2.2;                              // 挂在体积正下方
const PROJ_IDS = ['QKV_A', 'Q_B', 'O_A', 'O_B'];

const COL = {
  pending: new THREE.Color('#ccd6de'),
  done: new THREE.Color('#2f6fae'),
  active: new THREE.Color('#f4a23b'),
  idleSM: new THREE.Color('#e05a5a'),
  busySM: new THREE.Color('#f4a23b'),
  restSM: new THREE.Color('#dde4ea'),
  dim: new THREE.Color('#b9c6d1'),      // 未选中的体积
};

const state = {
  M: 896, tp: 1, sel: 'QKV_A',
  playing: true, speed: 190,  // speed = 每轮的毫秒数，和滑块 (280 - value) 对齐
  wave: 0, acc: 0,
};

/* ----------------------------------------------------------- three 基础设施 */
const canvas = $('gl');
// alpha: true，让 #stage 的浅色渐变透上来；否则画布会是不透明的黑底
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.background = null;

const camera = new THREE.PerspectiveCamera(42, 2, 0.5, 6000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.92;

scene.add(new THREE.AmbientLight(0xffffff, 1.5));
const key = new THREE.DirectionalLight(0xffffff, 2.0);
key.position.set(0.5, 1, 0.8);
scene.add(key);
const rim = new THREE.DirectionalLight(0xbdd4e8, 0.9);
rim.position.set(-0.6, 0.3, -0.9);
scene.add(rim);

const root = new THREE.Group();
scene.add(root);

const boxGeo = new THREE.BoxGeometry(1, 1, 1);
const tileMat = new THREE.MeshLambertMaterial();
const smMat = new THREE.MeshLambertMaterial();

let volumes = [];        // 每个投影一条 {p, t, mesh, x0, w, h, d, waveOf[]}
let smMesh = null;       // 152 个 SM 的小方块
let totalW = 0;
const dummy = new THREE.Object3D();

/* ----------------------------------------------------------------- 构建场景 */
function tileX(col, t, p) {
  // O_A 是 BMM，分组之间留缝，让 8 个独立矩阵看得出来
  return p.perGroupN ? col + Math.floor(col / t.colsPerGroup) * GROUP_GAP : col;
}
function volWidth(t, p) {
  return p.perGroupN ? t.cols + (p.groups - 1) * GROUP_GAP : t.cols;
}

function build() {
  // 只 dispose InstancedMesh 自己的 instance buffer；boxGeo / 材质是四个体积共用的，不能释放
  for (const v of volumes) { root.remove(v.mesh); v.mesh.dispose(); }
  volumes = [];
  if (smMesh) { root.remove(smMesh); smMesh.dispose(); }

  const ps = projections(state.M, state.tp);
  let x = 0;
  for (const p of ps) {
    const t = tiling(p);
    const w = volWidth(t, p), h = t.rows, d = p.K / TILE;
    const mesh = new THREE.InstancedMesh(boxGeo, tileMat, t.tiles);
    mesh.userData.id = p.id;
    let i = 0;
    for (let r = 0; r < t.rows; r++) {
      for (let c = 0; c < t.cols; c++) {
        dummy.position.set(x + tileX(c, t, p) + 0.5, h - r - 0.5, -d / 2);
        dummy.scale.set(0.88, 0.88, d);
        dummy.updateMatrix();
        mesh.setMatrixAt(i++, dummy.matrix);
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    root.add(mesh);
    volumes.push({ p, t, mesh, x0: x, w, h, d });
    x += w + GAP;
  }
  totalW = x - GAP;
  root.position.x = -totalW / 2;

  // 152 个 SM：19 × 8 的小方块阵，挂在选中体积下方
  smMesh = new THREE.InstancedMesh(boxGeo, smMat, SMS);
  for (let i = 0; i < SMS; i++) {
    const r = Math.floor(i / SM_COLS), c = i % SM_COLS;
    dummy.position.set(c * SM_PITCH, -r * SM_PITCH, 0);
    dummy.scale.set(0.9, 0.9, 0.9);
    dummy.updateMatrix();
    smMesh.setMatrixAt(i, dummy.matrix);
  }
  smMesh.instanceMatrix.needsUpdate = true;
  root.add(smMesh);

  buildLabels();
  state.wave = 0; state.acc = 0;
  placeSMGrid();
  paint();
}

/* ------------------------------------------------------------------- 上色 */
function paint() {
  const selV = volumes.find((v) => v.p.id === state.sel);
  for (const v of volumes) {
    const isSel = v === selV;
    for (let i = 0; i < v.t.tiles; i++) {
      let c;
      if (!isSel) {
        c = COL.dim;
      } else {
        const w = Math.floor(i / SMS);
        c = w === state.wave ? COL.active : w < state.wave ? COL.done : COL.pending;
      }
      v.mesh.setColorAt(i, c);
    }
    v.mesh.instanceColor.needsUpdate = true;
  }
  // SM 阵：本轮有多少个块，剩下的就是空转
  const inWave = selV ? (state.wave === selV.t.waves - 1 ? selV.t.lastWave : SMS) : 0;
  for (let i = 0; i < SMS; i++) smMesh.setColorAt(i, i < inWave ? COL.busySM : COL.idleSM);
  smMesh.instanceColor.needsUpdate = true;

  if (selV) {
    const pct = (inWave / SMS * 100).toFixed(1);
    $('waveText').innerHTML =
      `第 <b>${state.wave + 1}</b> / ${selV.t.waves} 轮　本轮 <b>${inWave}</b> / ${SMS} 个 SM（${pct}%）`;
    $('waveFill').style.width = ((state.wave + 1) / selV.t.waves * 100) + '%';
  }
}

function placeSMGrid() {
  const v = volumes.find((x) => x.p.id === state.sel);
  if (!v || !smMesh) return;
  smMesh.position.set(v.x0 + v.w / 2 - SM_W / 2, SM_TOP, 2.5);
}

/* ------------------------------------------------------------------- 标签 */
function buildLabels() {
  const host = $('labels');
  host.replaceChildren();
  for (const v of volumes) {
    const d = document.createElement('div');
    d.className = 'tag3d' + (v.p.id === state.sel ? ' sel' : '');
    d.innerHTML = `${v.p.label}<small>${v.t.cols}×${v.t.rows} 块 · ${v.t.waves} 轮</small>`;
    host.appendChild(d);
    v.label = d;
  }
  const mk = (t) => { const e = document.createElement('div'); e.className = 'ax'; e.textContent = t; host.appendChild(e); return e; };
  axN = mk(''); axM = mk(''); axK = mk(''); axSM = mk('');
}
let axN, axM, axK, axSM;

const _v = new THREE.Vector3();
function project(x, y, z) {
  _v.set(x, y, z).add(root.position).project(camera);
  return { x: (_v.x * 0.5 + 0.5) * canvas.clientWidth, y: (-_v.y * 0.5 + 0.5) * canvas.clientHeight, z: _v.z };
}
function putLabel(e, x, y, z) {
  const p = project(x, y, z);
  const vis = p.z < 1;
  e.style.display = vis ? '' : 'none';
  if (vis) { e.style.left = p.x + 'px'; e.style.top = p.y + 'px'; }
}
function updateLabels() {
  for (const v of volumes) putLabel(v.label, v.x0 + v.w / 2, v.h + 1.6, 0);
  const v = volumes.find((x) => x.p.id === state.sel);
  if (!v) return;
  // 全览时体积缩得很小，这几个轴标签会挤成一团，所以按屏幕上的实际宽度决定显不显示
  const onScreenW = Math.abs(project(v.x0 + v.w, v.h / 2, 0).x - project(v.x0, v.h / 2, 0).x);
  if (onScreenW < 70) {   // 全览时这个值只有 30 上下，聚焦时最窄的近端面也有 100 出头
    for (const e of [axN, axM, axK, axSM]) e.style.display = 'none';
    return;
  }
  axN.textContent = `N = ${fmt(v.p.N)}（${v.t.cols} 块宽）`;
  axM.textContent = `M = ${fmt(v.p.M)}（${v.t.rows} 块高）`;
  axK.textContent = `K = ${fmt(v.p.K)}（深 ${(v.d).toFixed(0)} 格）`;
  axSM.textContent = `GB200 的 ${SMS} 个 SM`;
  putLabel(axN, v.x0 + v.w / 2, -1.1, 0);
  putLabel(axM, v.x0 - 1.8, v.h / 2, 0);
  putLabel(axK, v.x0 + v.w + 1.2, v.h, -v.d / 2);
  putLabel(axSM, v.x0 + v.w / 2 + SM_W / 2 + 3.2, SM_TOP - SM_H / 2, 2.5);
}

/* ------------------------------------------------------------------- 相机 */
let camTarget = null;   // {pos, look, t}
function flyTo(pos, look) {
  camTarget = { pos: pos.clone(), look: look.clone(), t: 0 };
}
const DIR_OVER = new THREE.Vector3(0.10, 0.52, 0.85).normalize();
const smBottom = () => SM_TOP - SM_H;

// 固定斜视角。Q_B 宽 256 块、高只有 7 块，正着看会压成一条线；
// 斜着看反而把它变成一条退向远处的长廊，一轮一轮的推进看得清楚。
const DIR_FOCUS = new THREE.Vector3(0.34, 0.40, 0.85).normalize();

// 四边能用多少画幅（相对半宽 / 半高）。底部让给 HUD，左边让给读数面板，右边让给图例。
const FR = { left: 0.50, right: 0.88, top: 0.90, bottom: 0.58 };
const _up = new THREE.Vector3(0, 1, 0);
const _r = new THREE.Vector3(), _u = new THREE.Vector3(), _l = new THREE.Vector3();

/* 体积是斜着看的，光按 M/N 的跨度算距离会漏掉 K 方向伸出来的部分。
 * 这里把要看的角点逐个投到视锥坐标里，反推最少要退多远才全部落进可用画幅。 */
function flyFit(dir, pts) {
  const box = new THREE.Box3().setFromPoints(pts);
  const center = box.getCenter(new THREE.Vector3());
  _r.crossVectors(_up, dir).normalize();
  _u.crossVectors(dir, _r).normalize();
  const half = Math.tan((camera.fov * Math.PI) / 360), hw = half * camera.aspect;
  let dist = 1;
  for (const p of pts) {
    _l.copy(p).sub(center);
    const x = _l.dot(_r), y = _l.dot(_u), z = _l.dot(dir);
    dist = Math.max(dist,
      z + Math.abs(y) / (half * (y >= 0 ? FR.top : FR.bottom)),
      z + Math.abs(x) / (hw * (x >= 0 ? FR.right : FR.left)));
  }
  flyTo(center.clone().addScaledVector(dir, dist), center);
}

function cornersOf(v, withSM) {
  const x0 = v.x0 - totalW / 2, x1 = x0 + v.w, pts = [];
  for (const x of [x0, x1]) for (const y of [0, v.h]) for (const z of [-v.d, 0]) pts.push(new THREE.Vector3(x, y, z));
  if (withSM) {
    const gx = (x0 + x1) / 2;
    for (const x of [gx - SM_W / 2, gx + SM_W / 2]) for (const y of [smBottom(), SM_TOP]) pts.push(new THREE.Vector3(x, y, 2.5));
  }
  return pts;
}
function overview() {
  flyFit(DIR_OVER, volumes.flatMap((v) => cornersOf(v, v.p.id === state.sel)));
}
function focusOn(id) {
  const v = volumes.find((x) => x.p.id === id);
  if (v) flyFit(DIR_FOCUS, cornersOf(v, true));
}

/* ------------------------------------------------------------------- 读数 */
function readout() {
  const v = volumes.find((x) => x.p.id === state.sel);
  if (!v) return;
  const c = cost(v.p);
  $('roName').textContent = v.p.label;
  $('roExpr').textContent = v.p.expr.replace(/\bM\b/g, fmt(v.p.M));
  const empty = SMS * v.t.waves - v.t.tiles;
  const rows = [
    ['K（深度）', fmt(v.p.K)],
    ['N（宽度）', fmt(v.p.N)],
    ['块数', `${v.t.cols} × ${v.t.rows} = ${fmt(v.t.tiles)}`],
    ['轮数', String(v.t.waves)],
    ['末轮占用', (v.t.occupancy * 100).toFixed(1) + '%', v.t.occupancy < 0.75],
    ['空转 SM-轮次', fmt(empty), empty > v.t.tiles * 0.3],
    ['乘加量', si(c.flops, 'FLOP')],
    ['TP 切法', v.p.shard ? `切 ${v.p.shard}` : '不切'],
  ];
  $('roStats').innerHTML = rows.map(([k, val, hl]) =>
    `<dt>${k}</dt><dd class="${hl ? 'hl' : ''}">${val}</dd>`).join('');
}

/* ------------------------------------------------------------------- 交互 */
function segment(host, items, get, set) {
  host.replaceChildren();
  for (const [val, text] of items) {
    const b = document.createElement('button');
    b.textContent = text;
    b.className = get() === val ? 'on' : '';
    b.onclick = () => { set(val); for (const o of host.children) o.className = ''; b.className = 'on'; };
    host.appendChild(b);
  }
}

function selectProj(id) {
  state.sel = id;
  state.wave = 0; state.acc = 0;
  for (const v of volumes) v.label.className = 'tag3d' + (v.p.id === id ? ' sel' : '');
  placeSMGrid(); paint(); readout(); focusOn(id);
}

// 两组分段按钮在启动段统一建（那时 URL 参数已经读完），这里只挂其余控件
$('m').addEventListener('input', (e) => {
  state.M = +e.target.value;
  $('mOut').textContent = fmt(state.M);
  $('mRb').textContent = `= ${Math.ceil(state.M / TILE)} 块高`;
  build(); readout();
});
$('play').onclick = () => { state.playing = !state.playing; $('play').textContent = state.playing ? '暂停' : '播放'; };
$('speed').addEventListener('input', (e) => { state.speed = 280 - +e.target.value; });
$('overview').onclick = overview;
$('reset').onclick = () => focusOn(state.sel);

// 点体积切换
const ray = new THREE.Raycaster();
const ndc = new THREE.Vector2();
let downAt = null;
canvas.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
canvas.addEventListener('pointerup', (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
  const r = canvas.getBoundingClientRect();
  ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const hit = ray.intersectObjects(volumes.map((v) => v.mesh), false)[0];
  if (hit?.object?.userData?.id && hit.object.userData.id !== state.sel) {
    selectProj(hit.object.userData.id);
    for (const [i, b] of [...$('segProj').children].entries()) {
      b.className = PROJ_IDS[i] === state.sel ? 'on' : '';
    }
  }
});

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

/* ------------------------------------------------------------------- 主循环 */
let last = performance.now();
function frame(now) {
  const dt = Math.min(now - last, 100); last = now;
  resize();

  const selV = volumes.find((v) => v.p.id === state.sel);
  if (state.playing && selV) {
    state.acc += dt;
    if (state.acc >= state.speed) {
      state.acc = 0;
      state.wave = (state.wave + 1) % selV.t.waves;
      paint();
    }
  }

  if (camTarget) {
    camTarget.t = Math.min(camTarget.t + dt / 620, 1);
    const k = 1 - Math.pow(1 - camTarget.t, 3);        // ease-out
    camera.position.lerp(camTarget.pos, k * 0.22);
    controls.target.lerp(camTarget.look, k * 0.22);
    if (camTarget.t >= 1 && camera.position.distanceTo(camTarget.pos) < 0.4) camTarget = null;
  }

  controls.update();
  renderer.render(scene, camera);
  updateLabels();
  requestAnimationFrame(frame);
}

/* --------------------------------------------------------------------- 启动 */
const q = new URLSearchParams(location.search);
if (PROJ_IDS.includes(q.get('sel'))) state.sel = q.get('sel');
const qm = Number(q.get('m')); if (qm >= 64 && qm <= 2432) state.M = Math.round(qm / 64) * 64;
const qtp = Number(q.get('tp')); if ([1, 2, 4, 8].includes(qtp)) state.tp = qtp;
$('m').value = state.M;
$('mOut').textContent = fmt(state.M);
$('mRb').textContent = `= ${Math.ceil(state.M / TILE)} 块高`;
segment($('segProj'), PROJ_IDS.map((i) => [i, i]), () => state.sel, selectProj);
segment($('segTp'), [[1, 'DP'], [2, 'TP2'], [4, 'TP4'], [8, 'TP8']], () => state.tp,
  (v) => { state.tp = v; build(); readout(); focusOn(state.sel); });

build();
readout();
camera.position.set(40, 30, 90);
resize();              // 先拿到真实 aspect，再算取景距离
focusOn(state.sel);
requestAnimationFrame(frame);
void CFG;
