# Demo Dataset 结构 (data_sample_1000)

> 来源: https://huggingface.co/datasets/TAAC2026/data_sample_1000
> 文件: `demo_1000.parquet` (HF 路径: `hf://datasets/TAAC2026/data_sample_1000/demo_1000.parquet`)
> 实测加载方式 (二选一):
> ```python
> import pandas as pd
> df = pd.read_parquet("hf://datasets/TAAC2026/data_sample_1000/demo_1000.parquet")
> # 或
> from datasets import load_dataset
> ds = load_dataset("TAAC2026/data_sample_1000")
> ```
> 复现脚本:
> - `scripts/probe_demo_dataset.py` — 打印所有列的结构/类型/空值率
> - `scripts/show_one_sample.py [ROW_INDEX]` — 以人类可读格式展示一条完整样本 (默认 row 0)

**重要**: 正式比赛数据的结构/字段名与此可能不一致 (官方明确说可能调整)。此文档仅描述 demo。

## 0. 一条样本的逻辑结构 (三元组展开在 120 列中)

一条样本本质是 `(user, item, context) → label` 的三元组，但在 Parquet 里它被**展平为单行 120 列**，按前缀分组：

```
一条样本 (1 行 × 120 列)
│
├── [KEY / LABEL]  (5 列)   ← 不是特征，是主键和标签
│     user_id, item_id, label_type, label_time, timestamp
│
├── [USER 侧特征]
│     ├── user_int_feats_*    (46 列)  用户静态属性 (年龄/性别/婚姻/地域编号 …)
│     ├── user_dense_feats_*  (10 列)  预训练稠密向量 (_61=256 维, _87=320 维 …)
│     └── domain_a/b/c/d_seq_*  (9+14+12+10 = 45 列)  用户的历史行为序列
│          每个 domain 的 N 列是并列的 N 个等长数组，共同描述一条序列
│          例: domain_b: _67=时间戳, _68=action_type, _69=item_id, 其余=side info
│
└── [ITEM 侧特征]
      └── item_int_feats_*    (14 列)   目标广告属性 (类目/广告主/类型 …)
```

### 为什么 user 特征那么多、item 特征那么少？

CVR/CTR 建模的常态：
- **Item 侧**只描述"这一条广告本身" → 字段少 (14 列)
- **User 侧**要刻画一个人的兴趣 → 人口学属性 (user_int / user_dense) + 历史行为序列 (4 个 domain)，共 101 列

### 对应官方"统一建模"框架

```
序列特征 (domain_a/b/c/d_seq_*) ─▶ 序列 Tokenizer ─▶ S tokens ┐
                                                             ├─▶ 统一 Block × N ─▶ CVR Head
非序列特征 (user_int + user_dense + item_int) ─▶ NS Tokenizer ─▶ NS tokens ┘
```

- **S tokens** 来源 = 4 个 domain 的序列列
- **NS tokens** 来源 = `user_int_feats_*` + `user_dense_feats_*` + `item_int_feats_*`

> 官网 JSON 示例中的 **Context Features** (设备品牌/OS) 和 **Cross Features** (稠密交叉向量) 在 demo Parquet 中未单独分组，疑似已合并到 `user_int_feats_*` 与 `user_dense_feats_*` 内 (没有 `context_*` / `cross_*` 前缀)。正式数据是否新增前缀待 release 时确认。

### 快速查看一条样本

```bash
python3 scripts/show_one_sample.py        # 默认查看 row 0
python3 scripts/show_one_sample.py 42     # 查看第 42 条
```

该脚本会按 `Meta / User int / User dense / Item int / domain_a~d_seq` 分区打印，稀疏数组只显示前 10 项，稠密向量只显示前 6 维，并给出每个 domain 序列的本行长度。

## 1. 形状与主键

- **1000 行 × 120 列**
- 每行 = 一条样本 = (user, item, context, label)
- `user_id` 在 demo 中 **唯一** (每用户 1 条样本)

## 2. Meta / Label 字段

| 列 | dtype | 说明 (推测) |
|---|---|---|
| `user_id` | int64 | 用户 ID |
| `item_id` | int64 | 目标广告 ID (837 个 unique / 1000 行) |
| `label_type` | int32 | **标签**，取值 {1, 2}；分布 876 / 124 (推测 1=非转化, 2=转化, 正样本率 ≈12.4%) |
| `label_time` | int64 | 标签事件发生的 Unix 时间戳 (秒) |
| `timestamp` | int64 | 样本 (曝光?) 时间戳 (秒)；`label_time − timestamp ∈ [2, 832]s`，总在标签之前 |

> ⚠️ `label_type` 只有 2 种值但命名像类型而非 0/1；需在拿到正式数据后确认 label 编码语义。

## 3. 特征分组一览

| 组 | 列数 | 说明 |
|---|---|---|
| `user_int_feats_*` | 46 | 用户侧离散/整数特征 |
| `user_dense_feats_*` | 10 | 用户侧稠密 embedding 向量 |
| `item_int_feats_*` | 14 | 广告/物品侧离散/整数特征 |
| `domain_a_seq_*` | 9  | 行为序列域 A (并列 9 列) |
| `domain_b_seq_*` | 14 | 行为序列域 B (并列 14 列) |
| `domain_c_seq_*` | 12 | 行为序列域 C (并列 12 列) |
| `domain_d_seq_*` | 10 | 行为序列域 D (并列 10 列) |

**编号不连续但全局唯一**: 列名中的数字即原始的 `feature_id` (0–109 之间)，与官网示例一致。例如 `user_int_feats_7` / `user_int_feats_8` / `user_int_feats_10` 分别对应 Age / Gender / Marital Status。

## 4. 特征值类型规律 (parquet 表达)

| 原语义 | parquet dtype | 说明 |
|---|---|---|
| 单值整数 (`int_value`) | `int64` 或 `float64` (有 null 时) | 离散 ID |
| 多值整数 (`int_array`) | `object` (内含 `ndarray[int64]`) | 变长整数数组 |
| 稠密向量 (`float_array`) | `object` (内含 `ndarray[float32]`) | 定长 embedding |

## 5. 用户侧特征 (user_int_feats) 关键字段

已知语义 (对齐官网示例):

| 列 | 类型 | 样例 | 语义 |
|---|---|---|---|
| `user_int_feats_7`  | int_value | 44 | Age |
| `user_int_feats_8`  | int_value | 1  | Gender |
| `user_int_feats_10` | int_array | 见下 | Marital Status |

多值 (int_array) 列: `_15, _60, _62, _63, _64, _65, _66, _80, _89, _90, _91` (ndarray[int64]，变长)。
其中 `_89/_90/_91` 定长 = 10。

null 率差异大: 低缺失 (<5%): `_1,_3,_4,_48..53,_62..65`; 高缺失 (>50%): `_60,_86,_94,_96,_99..103,_109`。

## 6. 用户侧稠密向量 (user_dense_feats)

| 列 | 向量维度 | 说明 (推测) |
|---|---|---|
| `user_dense_feats_61` | **256** | 用户 embedding |
| `user_dense_feats_62` | 2  | 统计量 |
| `user_dense_feats_63` | 4  | 统计量 |
| `user_dense_feats_64` | 3  | 统计量 |
| `user_dense_feats_65` | 4  | 统计量 |
| `user_dense_feats_66` | 4  | 统计量 |
| `user_dense_feats_87` | **320** | 另一用户 embedding |
| `user_dense_feats_89` | 10 | |
| `user_dense_feats_90` | 10 | |
| `user_dense_feats_91` | 10 | (45% 缺失) |

全部 `float32`，所有向量在整列内 **定长**。

## 7. 物品侧特征 (item_int_feats)

14 列，基本为单值 (`float64` 带少量 null)；`_11` 为多值 int_array。
已知语义 (来自官网示例): `_7` = 广告类型 (Type, 官网为 feature_id 70 但 demo 重编到此处需验证), `_60` (位于 user 组) = Category, `_72` = Advertiser Type.

> ⚠️ demo 中的 feature_id 编号与官网示例的编号在部分字段上 **不完全对应**，以实际列名为准。

## 8. 行为序列 (domain_a/b/c/d)

每个 domain 的 N 个列对应 **同一条序列的 N 个并列字段** (时间戳 / 事件类型 / item_id / side info 等)，在同一行内 **长度完全一致**。不同 domain 彼此独立，长度不同。

### 长度分布 (1000 行, 基于各 domain 第 1 列):

| domain | min | p50 | p95 | max | mean |
|---|---|---|---|---|---|
| a | 0 | 577  | 1673 | 1888 | 701.1 |
| b | 0 | 405  | 1563 | 1952 | 570.8 |
| c | 0 | 322  | 1214 | 3894 | 449.4 |
| d | 0 | 1035 | 2451 | 3951 | 1099.9 |

### 典型字段含义 (从样例数值推断)

- `domain_a_seq_38`: item-like ID
- `domain_a_seq_39`: timestamp (1.77e9, Unix 秒)
- `domain_a_seq_40/41`: action/type 类小整数 (值域个位数)
- `domain_a_seq_42..46`: side info (类别/统计)
- `domain_b_seq_67`: timestamp
- `domain_b_seq_68`: action_type
- `domain_b_seq_69`: large ID (可能 item_id)
- `domain_c_seq_27`: timestamp
- `domain_c_seq_47`: 大 ID (推测 item_id)
- `domain_d_seq_26`: timestamp
- `domain_d_seq_17..25`: ID/side info

> domain 内每条列都是 **ndarray[int64]**。时间戳列可用于对序列排序/截断。

## 9. 缺失处理

- 单值列用 `None` / NaN (dtype 被提升为 float64)
- 多值 / 稠密列当缺失时整格为 Python `None`
- 序列列当缺失时整格为 `None` (表示该用户在该 domain 无行为)

## 10. 代码加载提示

- Parquet 中 `object` 列的元素是 `numpy.ndarray`；在 pytorch 侧需自行 padding / truncation。
- Dense embedding 为 `float32`，可直接 `torch.from_numpy(arr)`。
- 推荐先按 `timestamp` 过滤/排序序列 (取样本 timestamp 之前的行为)。

## 11. 与官方描述的差异 / TODO

- [ ] demo 列名中的 feature_id 与官网 JSON 示例存在部分对不上的地方，等正式数据 schema 补齐
- [ ] 确认 `label_type` 取值 (1/2) 的具体语义：是否 2 = 转化、1 = 曝光未转化？
- [ ] 确认每个 domain 9~14 列分别对应 item_id / action_type / timestamp / 各 side info 的准确映射
- [ ] 训练/验证切分规则 (预计按 `label_time` 时间切)
- [ ] 推理延迟计分细节
