# MQuAKE-T 补充实验记录

本文件用于记录时间更新数据集 `MQuAKE-T` 上的补充实验状态与结论。

## 已完成尝试

### 1. 严格过滤

过滤条件：

1. 单跳旧事实生成正确；
2. 多跳旧事实生成正确；
3. 旧答案打分高于新答案。

结果：

- `Qwen2.5-0.5B-Instruct`：样本池为 `0`
- `Qwen2.5-1.5B-Instruct`：样本池为 `0`
- `Qwen2.5-3B-Instruct`：样本池为 `0`
- `Qwen2.5-7B-Instruct`：样本池为 `0`

初步结论：

`MQuAKE-T` 对底模的现实时间知识与多跳链路要求明显高于 `MQuAKE-CF`。在当前可用模型中，即便单跳常常能答对旧事实，也很难同时满足严格的多跳旧答案偏好条件。

### 2. Generation Filter 补充尝试

过滤条件：

1. 单跳旧事实生成正确；
2. 多跳旧事实生成正确。

结果：

- `Qwen2.5-3B-Instruct` 在前 `240` 个样本中可筛出 `20` 个样本；
- 但 CQAG 在该设置下未带来有效提升：
  - `multihop_no_cqag_new_acc = 0.000`
  - `multihop_cqag_full_new_acc = 0.000`
  - `locality = 0.929`

### 3. 7B 扩展扫描结果

设置：

- 模型：`Qwen2.5-7B-Instruct`
- 加载方式：`4bit`
- 过滤方式：`generation filter`
- 扫描范围：前 `800` 个时间样本
- 目标池大小：`12`

结果：

- 成功构造最小实验池：`12`
- 校准结果：
  - 最优层：`22`
  - 最优 `alpha`：`2.5`
- 测试集结果：
  - `multihop_no_cqag_new_acc = 0.000`
  - `multihop_cqag_full_new_acc = 0.500`
  - `multihop_cqag_fact_only_new_acc = 0.000`
  - `multihop_cqag_random_new_acc = 0.250`
  - `multihop_base_old_pref_acc = 0.875`
  - `locality_mismatched_old_pref_acc = 0.250`

结论：

1. 时间更新实验在更强模型与更大扫描范围下终于出现正向提升；
2. `CQAG-full` 能将时间集多跳新答案成功率从 `0.000` 提升到 `0.500`；
3. 但随机向量也有 `0.250` 的命中，且局部性仅为 `0.250`，说明时间更新场景比反事实场景更不稳定，干预更容易带来副作用；
4. 因而时间集结果更适合被表述为“初步有效但稳定性不足”。
