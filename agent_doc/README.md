# agent_doc

Agent 记录文件夹，供后续任务阅读与断点恢复使用。

## 文件索引

- `01_competition_overview.md` — TAAC2026 / KDD Cup 2026 赛事总览 (背景、目标、赛道、奖金、时间线、规则)
- `02_data_format.md` — 官方 JSON 示例的数据格式说明 (序列 / 用户 / 物品 / 上下文 / 交叉特征；已知 feature_id 对照表)
- `03_demo_dataset.md` — **实测 demo Parquet 数据集结构** (1000 × 120，4 个序列域，稠密 embedding 维度等)

相关脚本:
- `scripts/probe_demo_dataset.py` — 复现 demo 数据结构探测

## 关键信息速查

- 赛题: Tencent UNI-REC Challenge (统一序列建模 + 特征交互)
- 任务: pCVR 预测，单一指标 AUC of ROC
- 硬约束: 推理延迟上限 + 禁止 ensemble
- 数据: 匿名化整数 ID + dense embedding；JSON(官网示例) / Parquet(demo 实物)
- Demo 实测: 1000 行 × 120 列，含 4 个行为序列域 (a/b/c/d)，序列长度 p95 最高可达 2451，用户 embedding 维度 256 与 320
- 时间: 2026-03-15 发布 demo → 03-19~04-23 报名 → 04-24~05-23 Round1 → 05-25~06-24 Round2 → 07-15 通知 → 08-09 KDD 颁奖
- 奖金池: $885,000，学术赛道冠军 $300,000

## 维护约定

- 拿到正式数据后需回填 `02_data_format.md` 底部的 TODO 列表
- 重要决策 / 实验记录按日期新增 `NN_xxx.md` 文件，保持编号递增
