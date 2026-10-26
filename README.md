# V4.1-Flash Attention 结构可视化

一个交互页面：拖动每卡 query 行数 **M**，看四个投影的输出怎么切成 128×128 的方块、
占掉 GB200 152 个 SM 里的多少个、需要几轮；切换 **DP / TP2 / TP4 / TP8**，
看每张卡实际拿到多大的矩阵；点数据流里任意一步，看它具体的矩阵乘法。

纯静态页面，没有依赖，没有构建步骤。

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

- `m` —— 每卡 query 行数，64–2432，64 的倍数
- `tp` —— 1 / 2 / 4 / 8，1 就是 Attention DP
- `sel` —— `QKV_A` / `Q_B` / `O_A` / `O_B`
- `ctx` —— 上下文长度的以 2 为底的指数，10–20

## 几个值得拖一下的地方

- **M=896 → M=1408，看 QKV_A。** 方块从 98 涨到 154，刚好越过 152，末轮占用掉到 1%。
  这就是 wave 取整的悬崖。
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

瓦片与 wave 是形状算术：`块数 = ⌈N/128⌉ × ⌈M/128⌉`，`轮数 = ⌈块数 / 152⌉`。

## 这里没有的东西

- **没有任何实测时间、吞吐或 TPOT。** 整个页面只有 config 维度和形状算术。
- 瓦片和 wave 是理想排布，不含 L2、尾效应、graph padding。
- wave 数假设每个方块代价相同，实际不是 —— 每块的活跟 K 成正比，
  `Q_B` 的 K 只有 1280，块多但每块很浅。所以**轮数不能直接当时间读**。
- `O_A` 真实 kernel 的瓦片不是 128×128，这里按统一口径算，只用于比较量级。
- 只画 attention。MoE、mHC、shared expert、Engram 表都没有画。

## 文件

```
index.html   结构
style.css    样式
model.js     config 常量 + 形状/瓦片/字节推导（无 DOM，可单独跑）
app.js       渲染与交互
```

`model.js` 不依赖浏览器，可以直接用 node 导入来核对推导。
