# 腾讯算法大赛 2026 (TAAC x KDD Cup 2026) 概览

> 来源: https://algo.qq.com/ (TAAC2026 / Tencent UNI-REC Challenge)
> 记录时间: 2026-04-23

## 1. 赛事身份

- 全称: **Tencent UNI-REC Challenge** (TAAC x KDD Cup 2026)
- 主题: **Towards Unifying Sequence Modeling and Feature Interaction for Large-scale Recommendation**
- 级别: KDD Cup 2026 官方赛题之一 (KDD 是数据挖掘领域与 ICML、NeurIPS 并列的三大顶会)
- 官方邮箱: taac@tencent.com
- 官网: https://algo.qq.com/
- Demo 数据集: Hugging Face `TAAC2026/data_sample_1000` (约 1000 条样本，68MB，Parquet 格式)

## 2. 赛题核心命题

推荐系统中历来存在两个割裂的分支：
1. **特征交互 (Feature Interaction) 模型**: 处理高维稀疏多域特征 (DeepFM、DCN 等)
2. **序列建模 (Sequence Modeling) 模型**: 捕捉用户行为序列中的兴趣演变 (DIN、SASRec 等)

工业场景中这两类模块往往"各自为政"，导致:
- 架构碎片化，GPU 算力难以高效利用
- 优化目标 / 部署流程 / 调参逻辑各成两套
- 长行为序列 + 大模型规模下扩展性受限，无法系统性探索推荐系统的 **Scaling Law**

**赛题目标**: 设计一个 **单一同构、可堆叠的 Recommendation Block**，在同一架构中联合建模 **序列行为** 和 **非序列多域特征**，输出目标广告的 **pCVR (预测转化率)**。

## 3. 任务形式

- 每条训练/测试样本为一个 **三元组**: (用户 user, 上下文 context, 目标广告/商品 item)
- 输入:
  - **非序列多域特征**: 用户属性、广告属性、上下文特征、交叉特征 (cross features, dense embedding)
  - **用户行为序列**: 按时间排列的历史行为 (item_id, action_type, timestamp, 以及附带的异构 side info)
- 输出: 该目标广告的 pCVR
- 训练损失: **交叉熵 (Cross Entropy)**

官方推荐建模范式 (非强制):
```
seq features    ──▶ Sequence Tokenizer   ──▶ S tokens  ┐
                                                        ├─▶ 堆叠的统一 Block (同构 backbone) ─▶ CVR Head
non-seq features ─▶ Non-Seq Tokenizer    ──▶ NS tokens ┘
```

## 4. 评估指标

- **主排名指标**: AUC of ROC (单一指标，越高越好)
- **延迟约束**: 每个提交必须满足对应赛道/轮次的 **推理延迟上限**，超时提交直接作废，不论 AUC 多高
- **硬规则**: **全程禁止模型集成 / 融合 (Ensemble)** — 强制单一架构做到极致
- 第二轮数据量为第一轮的 **10 倍**，考验 scalability

## 5. 赛道设置

| 赛道 | 对象 | 总奖金 |
|---|---|---|
| Academic Track | 全球全日制学生 (本/硕/博/博士后) | $540,000 |
| Industrial Track | 所有社会人士 / 研究机构 / 企业 (首次开设) | $255,000 |

> 腾讯员工不能参加任一赛道。

### Academic Track 奖金
- 冠军 (1): $300,000 (约 200 万 RMB)
- 亚军 (1): $90,000
- 季军 (1): $45,000
- 4-10 名 (7): 每队 $15,000

### 独立创新奖 (不依赖排名)
- **Unified Architecture Innovation Award**: $45,000
- **Scaling Law Innovation Award**: $45,000
- 评审看方法论的 **新颖性与洞察深度**，而非 AUC

### 总奖金池
**$885,000**

## 6. 赛程时间线 (AOE 时间)

| 阶段 | 时间 | 事件 |
|---|---|---|
| Phase 1 | Mar.15, 2026 | Demo 数据集发布 |
| Phase 2 | Mar.19 — Apr.23 | 全球报名期 |
| Phase 3 | Apr.24 — May 23 | 第一轮比赛 |
| Phase 4 | May 25 — Jun.24 | 第二轮比赛 (数据量 x10) |
| Phase 5 | Jul.15, 2026 | 获奖通知 |
| —       | Aug. 9, 2026 | KDD 2026 现场正式公布 (济州岛) |

## 7. 参赛规则要点

- 每队 1-3 人，每人只能加入一支队伍
- **4月23日之后不允许任何队伍变更**
- 数据集匿名化，无原始文本/图片/PII
- 禁止模型 ensemble / fusion
- 官方提供 Angel 机器学习平台免费 GPU 算力

## 8. 本仓库定位

- 本仓库 (`rec_model_game`) 是参加该比赛的代码库
- 后续需要围绕:
  1. 数据加载 (Parquet / JSON)
  2. Tokenizer 设计 (seq / non-seq 统一表征)
  3. 统一同构 Block 骨干网络
  4. CVR head + AUC 评估 + 延迟优化
  5. 不得做 ensemble
