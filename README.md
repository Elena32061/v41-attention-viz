# V4.1-Flash Attention 结构可视化

两个页面，同一套数据：

- **`index.html` —— WebGL 3D 动画版。** 一次 GEMM 画成一个体积：
  1 个格子 = 一个 128×128 的输出块，所以宽是 `N/128`、高是 `M/128`，
  **往里的深度就是 `K/128`**。输出面天然就是块网格，K 变成看得见的深度。
  动画一轮一轮地刷过去（橙 = 本轮在算，蓝 = 已算完，灰 = 还没轮到），
  下面挂一排 152 个 SM 方块跟着同步，本轮没派到活的变红。
- **`2d.html` —— 2D 精确版。** 带完整数表：四个投影的逐项维度、字节数、
  运算强度，以及稀疏两级漏斗。要看准确数字看这个。

- **`dots3.html` —— dots3 的 2D 精确版。** 同一套面板、同一套分块口径，数据来自 dots3 的
  `config.json`（`model_dots3.js`），多一个 full / SWA 层类型开关，多一栏和 V4.1-Flash 并排。
  画的是 decode 的吸收（absorbed MLA）路径：QKV_A → Q_B → W_KC → core attention → W_VC → O。
  URL 多一个 `layer=full|swa`，`sel` 可取 `QKV_A / Q_B / IDX_Q / W_KC / W_VC / O`。
  校验：`node check_dots3.mjs`。

- **`kernels.html` —— 8 卡 B64 实测 kernel 清单。** 不是形状算术，是 nsys trace 切成 100 个
  decode step 后的逐 kernel 统计：每步概况、一步的执行顺序条、按功能汇总、逐 kernel 表。
  两个数据集，页面顶部可切：

  | 数据集 | SGLang | DeepInfer | 调度重叠 | 关键路径 |
  |---|---|---|---|---|
  | `aligned`（默认） | `n1_sg_b64_ovl_nsys_a1`（10-25） | `t1_di_b64_nsys_a1`（10-26） | 两侧都开 | 有 |
  | `legacy` | `e2_sg8_b64_extension_01`（10-23） | `e2_di8_b64_v3_...`（10-22） | SGLang 关 | 无 |

  对齐版两份 trace 在同一 Trial 2266850 的同两个节点上，配置对齐（Attention DP8 / MoE EP8、CP1、
  16K/512、全局 batch 64、不开投机），数据在 `kernels_b64_8card_aligned.json`，由
  `outputs/sg_di_kernel_compare_aligned_20261026/stage/build_kernels_json.py` 生成。
  旧版两份不同天、配置未对齐，在 mentor 的 qs8 上采的，只保留做对照。

  对齐版的主时间列是**关键路径 ms/步**（该 kernel 占住这一步墙钟的时长，逐行加上空隙正好等于步长）。
  累计 ms 作为次列保留：SGLang 并发重叠多，它的累计 13.62 ms 比步长 8.30 ms 还高，
  跨框架不能比累计。旧版只有累计列。

  URL：`?ds=aligned|legacy&fw=both|SGLang|DeepInfer&sort=crit|ms|count|us|bucket&hide=<bucket,...>`。
  校验：`node check_kernels.mjs`。

四页都是纯静态，没有构建步骤。3D 页把 three.js 放在 `vendor/` 里，运行时不连 CDN。

## 本地打开

因为用了 ES module，`file://` 直接打开会被 CORS 拦掉，需要起一个本地服务：

```sh
python3 -m http.server 8000
# 然后打开 http://localhost:8000
```

## 分享某一个具体配置

状态写在 URL 里，直接复制地址栏就能把当前这一屏发给别人：

```
?m=896&tp=4&sel=Q_B&ctx=14
```

- `m` —— 每卡 query 行数，64–2432，64 的倍数（两页通用）
- `tp` —— 1 / 2 / 4 / 8，1 就是 Attention DP（两页通用）
- `sel` —— `QKV_A` / `Q_B` / `O_A` / `O_B`（两页通用）
- `ctx` —— 上下文长度的以 2 为底的指数，10–20（只有 2D 页有）

## 几个值得拖一下的地方

- **M=896 → M=1408，看 QKV_A。** 方块从 98 涨到 154，刚好越过 152，末轮占用掉到 1.3%，
  空转 150 个 SM-轮次。这就是轮数取整的悬崖（wave quantization）。
  3D 页里第二轮只剩两个橙格子，旁边 150 个 SM 全红。
- **3D 页里比 QKV_A 和 Q_B 的形状。** QKV_A 是 14×7 的短块但深 40 格，
  Q_B 是 256×7 的长条却只深 10 格 —— 块数差 18 倍，每块的活差 4 倍。
  这是把"轮数不能直接当时间读"画出来，而不是写在注脚里。
- **把并行方式从 DP 切到 TP4。** 同样 M=896，四个投影合计从 18 轮掉到 7 轮 ——
  但那是每张卡的轮数，四张卡要各跑一遍，而且 O_B 之后还要 all-reduce。
- **上下文从 16K 拉到 1024K，看 E 栏。** 候选集在 16,384 封顶，
  所以第二级的工作量和上下文长度脱钩。

## 数字从哪来

`model.js` 里 `CFG` 的每一项都抄自 checkpoint 的 `config.json → text_config`，
字段名原样保留，没有换算。下面这些推导来自实现源码：

| 推导 | 依据 |
|---|---|
| `QKV_A` 输出 `1792 = 1280 + 512` | q 潜向量 `q_lora_rank`，KV 潜向量 = `num_key_value_heads × head_dim` |
| `Q_B` 输出 `32768 = 64 × 512` | `num_attention_heads × head_dim`，column-parallel |
| `O_A` 每组输入 `4096 = 64×512÷8` | `o_groups=8` 的 BMM，每组 `→ o_lora_rank` |
| `O_B` 的 `K = 8192 = 8×1024` | `o_groups × o_lora_rank`，row-parallel（切 K 不切 N） |
| 稀疏两级 | `sparse_stage=2`；第一级 `candidate_topk_blocks=2048 × candidate_block_size=8`；第二级 MQA indexer 选 `index_topk=512` 个 **token**（源码 `self.topk_tokens = cfg.index_topk`，单位是 token 不是 block） |

块数与轮数是形状算术：`块数 = ⌈N/128⌉ × ⌈M/128⌉`，`轮数 = ⌈块数 / 152⌉`。

## 这里没有的东西

- **没有任何实测时间、吞吐或 TPOT。** 整个页面只有 config 维度和形状算术。
  3D 动画的节奏是固定速度（滑块可调），**不代表任何真实 kernel 耗时**；
  一轮一轮按块号顺序点亮，也不是真实的 CTA 调度顺序。
- 块数和轮数是理想排布，不含 L2、尾效应、graph padding。
- 轮数假设每个块代价相同，实际不是 —— 每块的活跟 K 成正比，
  `Q_B` 的 K 只有 1280，块多但每块很浅。所以**轮数不能直接当时间读**。
- `O_A` 真实 kernel 的 tile 不是 128×128，这里按统一口径算，只用于比较量级。
- 只画 attention。MoE、mHC、shared expert、Engram 表都没有画。

## 文件

```
model.js     config 常量 + 形状/分块/字节推导（无 DOM，两页共用的唯一真源）
check.mjs    29 条断言，和静态图 attention_figure.py 对齐（node check.mjs）
style.css    两页共用的基础样式

index.html   3D 页结构
scene.css    3D 页样式（画布、浮层 HUD、投影到屏幕的标签）
scene.js     three.js 场景、分块体积、轮次动画、取景

2d.html      2D 页结构
app.js       2D 页的 SVG 渲染与交互

vendor/      three.js r0.186.1，随仓库发布，运行时不连 CDN
```

`model.js` 不依赖浏览器，可以直接用 node 导入来核对推导。`check.mjs` 把它和
`outputs/v41_attention_figure_20261026/` 的静态图锁在一起：M=896 时 DP 合计 18 轮、
TP4 本地合计 7 轮，两处对不上就会报错。

`vendor/` 里三个文件的 SHA256：

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `three.module.js` | 662,772 | `9052042d676cb0fdc1ddfefe193053f34b7ac0513a616fdac4535d49987812ea` |
| `three.core.js` | 1,458,113 | `9edde002b066a9a05676a6127f67735b62baf399bdea529f2f7e31657da769e6` |
| `OrbitControls.js` | 40,755 | `3d79d07ecb686b4e5d93232eedab255331c1beef711e13164eaa1f68655a5f2b` |

r186 把构建拆成了两个文件，`three.module.js` 会 import 同目录的 `three.core.js`，
两个都要在才能加载。
