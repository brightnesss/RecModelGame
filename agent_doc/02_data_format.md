# 数据格式说明 (TAAC2026 Uni-Rec Challenge)

> 注意: 官方说明数据目前提供的是示例 JSON 样本，**正式比赛中格式与内容可能调整**。Demo dataset 为 Parquet 格式。

## 1. 数据规模

- 真实腾讯广告日志构建，日服务数十亿用户
- 包含 **100+ 个脱敏特征字段**，覆盖:
  - 用户行为序列 (sequential)
  - 用户属性 (user non-seq)
  - 广告/物品属性 (item non-seq)
  - 上下文信号 (context)
  - 交叉特征 (cross features, dense)
- 第二轮数据量是第一轮的 10 倍

## 2. 特征值类型 (feature_value_type)

| 类型 | 说明 | 字段 |
|---|---|---|
| `int_value` | 单个离散整数 ID | `int_value` |
| `int_array` | 离散整数 ID 的数组 (多值稀疏) | `int_array` |
| `float_array` | 定长浮点向量 (dense embedding) | `float_array` |

所有稀疏特征均为 **匿名化整数 ID**，无原始文本/图像/URL/PII。

## 3. 数据组成

### 3.1 Sequential Data (用户行为序列)

一个 user 对应一条序列，每个行为事件是一个三元组 (item_id, action_type, timestamp)，还可能附带异构 side information。

```json
{
  "user_id": "1",
  "seq": [
    {"item_id": 16612, "action_type": 1, "timestamp": 1770564000},
    {"item_id": 49638, "action_type": 1, "timestamp": 1770564000},
    {"item_id": 173346, "action_type": 3, "timestamp": 1766960100},
    {"item_id": 49495, "action_type": 2, "timestamp": 1766576760},
    {"item_id": 1753, "action_type": 4, "timestamp": 1766399880}
  ]
}
```

- `action_type`: 行为类型 (如 曝光=1 / 点击 / 转化 等, 具体编码官方未公开)
- `timestamp`: Unix 时间戳

### 3.2 User Features (用户非序列特征)

```json
[
  {"feature_id": 10, "feature_value_type": "int_array", "int_array": [ ... ]},  // Marital Status (婚姻状态)
  {"feature_id": 8,  "feature_value_type": "int_value", "int_value": 1},        // Gender (性别)
  {"feature_id": 7,  "feature_value_type": "int_value", "int_value": 44}        // Age (年龄)
]
```

### 3.3 Item Features (广告/物品特征)

```json
[
  {"feature_id": 70, "feature_value_type": "int_value", "int_value": 2},  // Type (广告类型)
  {"feature_id": 60, "feature_value_type": "int_value", "int_value": 3},  // Category (品类)
  {"feature_id": 72, "feature_value_type": "int_value", "int_value": 2}   // Advertiser Type (广告主类型)
]
```

### 3.4 Context Features (上下文/会话特征)

```json
[
  {"feature_id": 17, "feature_value_type": "int_value", "int_value": 3},  // Device Brand (设备品牌)
  {"feature_id": 21, "feature_value_type": "int_value", "int_value": 3}   // OS Type (操作系统)
]
```

### 3.5 Cross Features (交叉特征, dense embedding)

```json
[
  {
    "feature_id": 25,
    "feature_value_type": "float_array",
    "float_array": [0.111, 0.057, 0.121, 0.043, -0.066, 0.081, 0.038, 0.105, -0.026, ...]
  }
]
```
> 如 "User Embedding" 这类预训练稠密向量。

## 4. 已知 feature_id 对照表 (来自官方示例)

| feature_id | 含义 | 类型 | 归属 |
|---|---|---|---|
| 7  | Age (年龄)              | int_value   | User |
| 8  | Gender (性别)           | int_value   | User |
| 10 | Marital Status (婚姻)   | int_array   | User |
| 17 | Device Brand (设备品牌) | int_value   | Context |
| 21 | OS Type (操作系统)      | int_value   | Context |
| 25 | User Embedding          | float_array | Cross |
| 60 | Category (品类)         | int_value   | Item |
| 70 | Type (广告类型)         | int_value   | Item |
| 72 | Advertiser Type         | int_value   | Item |

> 其余 feature_id 的具体语义在正式数据集文档中给出，需在拿到数据后补全。

## 5. 加载建议

- Demo: `TAAC2026/data_sample_1000` on Hugging Face
  - Parquet 格式，支持 `pandas` / `datasets` 库直接加载
- 正式比赛数据预计结构类似，字段可能有调整
- 样本标签为转化 label (0/1)，用于 CVR 预测 (具体 label 字段需拿到正式数据后确认)

## 6. 待确认事项 (TODO，等正式数据释出)

- [ ] 完整 feature_id -> 语义 映射表
- [ ] action_type 编码 (曝光 / 点击 / 转化 分别对应哪些数字)
- [ ] label 字段名与形式
- [ ] 训练 / 测试切分方式 (时间切分?)
- [ ] 序列长度上限、padding 规则
- [ ] Cross features 各 embedding 维度
- [ ] 推理延迟预算的具体数值 (两赛道 / 两轮各不同)
