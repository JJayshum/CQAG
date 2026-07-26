# CQAG 论文实验总总结

## 1. 实验资源与镜像

本轮实验已按中国网络环境改为镜像下载：

- 模型镜像：`ModelScope`
- 成功下载模型：
  - `Qwen/Qwen2.5-0.5B-Instruct`
  - `Qwen/Qwen2.5-1.5B-Instruct`
- 数据获取：
  - `MQuAKE-CF-3k-v2.json`
  - `MQuAKE-T.json`

本地关键路径：

- 模型：
  - `/root/autodl-tmp/modelscope_cache/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master`
  - `/root/autodl-tmp/modelscope_cache/models/Qwen--Qwen2.5-1.5B-Instruct/snapshots/master`
- 数据：
  - `/root/cqag_experiment/data/MQuAKE-CF-3k-v2.json`
  - `/root/cqag_experiment/data/MQuAKE-T.json`

## 2. 已完成实验

### 2.1 合成机制验证实验

脚本：

- `/root/cqag_experiment/run_synthetic_cqag.py`
- `/root/cqag_experiment/run_synthetic_multiseed.py`

结果：

- 单跳编辑后新事实准确率：`1.000`
- 单跳局部性：`1.000`
- 无 CQAG 时多跳新事实准确率：`0.000`
- 无 CQAG 时多跳旧事实准确率：`1.000`
- CQAG 后多跳新事实准确率：`1.000`
- CQAG 后多跳旧事实准确率：`0.000`
- 随机向量注入平均准确率：`0.0625`

多随机种子聚合结果见：

- `/root/cqag_experiment/results/synth_multiseed/aggregate.json`

结论：

1. 单跳编辑成功并不会自动带来多跳推理更新；
2. 多跳推理会稳定沿用旧桥接事实；
3. 对齐后的 CQAG 唤醒向量能够稳定修复推导链；
4. 该现象在 5 个随机种子上稳定成立。

### 2.2 真实模型 + MQuAKE-CF 反事实实验

脚本：

- `/root/cqag_experiment/run_real_model_probe.py`
- `/root/cqag_experiment/run_real_cqag_experiment.py`

主结果文件：

- `/root/cqag_experiment/results/real_cqag_cf.json`

#### 真实模型探测

在 `Qwen2.5-0.5B-Instruct` 上，对 `MQuAKE-CF-3k-v2` 前 40 个样本进行旧知识探测：

- 单跳旧事实生成准确率：`0.375`
- 多跳旧事实生成准确率：`0.425`
- 单跳和多跳同时正确：`0.175`

说明该模型虽然不强，但足以筛出一批“确实拥有旧知识链”的可评测样本。

#### CQAG 主实验

实验设置：

1. 从前 240 个样本中筛选出同时满足以下条件的高质量案例：
   - 单跳旧事实生成正确
   - 多跳旧事实生成正确
   - 候选答案打分中旧答案优于新答案
2. 得到 21 个高质量样本；
3. 用其中 7 个做校准，14 个做测试；
4. 在 Qwen 中间层做隐藏状态注入，搜索 `layer` 和 `alpha`。

校准结果：

- 最优层：`22`
- 最优注入强度：`2.5`

测试集结果：

- `edit_memory_singlehop_new_acc = 1.000`
- `multihop_no_cqag_new_acc = 0.000`
- `multihop_base_old_pref_acc = 0.929`
- `multihop_cqag_full_new_acc = 0.571`
- `multihop_cqag_fact_only_new_acc = 0.286`
- `multihop_cqag_random_new_acc = 0.000`
- `locality_mismatched_old_pref_acc = 1.000`

结论：

1. 在真实模型上，单跳编辑记忆可以直接提供新答案，但多跳问题在无引导时仍完全失败；
2. `CQAG-full` 将多跳新答案成功率从 `0.000` 提升到 `0.571`；
3. 仅使用事实子图方向而不加入答案方向时，效果下降到 `0.286`；
4. 随机向量完全无效，说明效果并非来自任意扰动；
5. 错配向量的局部性保持为 `1.000`，说明注入具有较强方向选择性。

### 2.3 时间更新集实验

尝试模型：

- `Qwen2.5-0.5B-Instruct`
- `Qwen2.5-1.5B-Instruct`

数据：

- `MQuAKE-T`

严格过滤结果：

在前 `240` 个样本中，`0.5B / 1.5B / 3B / 7B` 均无法筛出满足“旧单跳正确 + 旧多跳正确 + 旧答案打分优于新答案”的足够样本，因此严格版时间更新评测无法直接建立。

补充方案：

采用 `generation filter`，即仅要求：

1. 单跳旧事实生成正确；
2. 多跳旧事实生成正确。

#### 3B 补充实验

- 数据：`MQuAKE-T` 前 `240` 个样本
- 模型：`Qwen2.5-3B-Instruct (4bit)`
- 结果：
  - `multihop_no_cqag_new_acc = 0.000`
  - `multihop_cqag_full_new_acc = 0.000`
  - `locality = 0.929`

说明：

3B 模型虽然能筛出一定数量的时间样本，但 CQAG 尚未在该规模下带来有效提升。

#### 7B 补充实验

- 数据：`MQuAKE-T` 前 `800` 个样本
- 模型：`Qwen2.5-7B-Instruct (4bit)`
- 成功构造实验池：`12`
- 校准结果：
  - 最优层：`22`
  - 最优 `alpha`：`2.5`
- 测试结果：
  - `multihop_no_cqag_new_acc = 0.000`
  - `multihop_cqag_full_new_acc = 0.500`
  - `multihop_cqag_fact_only_new_acc = 0.000`
  - `multihop_cqag_random_new_acc = 0.250`
  - `multihop_base_old_pref_acc = 0.875`
  - `locality_mismatched_old_pref_acc = 0.250`

结论：

1. 时间更新集在更强模型和更大扫描范围下可以跑出正向结果；
2. `CQAG-full` 将多跳新答案成功率从 `0.000` 提升到 `0.500`；
3. 但时间更新场景稳定性明显弱于反事实场景，随机向量也会带来一定扰动，局部性保持较差；
4. 因此这部分结果适合在论文中表述为“初步有效，但仍存在显著稳定性挑战”。

## 3. 当前实验结论

综合合成实验与真实模型实验，可以得到较强的一致结论：

1. 知识编辑中的关键问题确实不是“模型是否知道新事实”，而是“模型能否在多步链路中调用新事实”；
2. 单跳编辑成功与多跳推理成功之间存在明显鸿沟；
3. CQAG 通过中间层隐藏状态注入，可以在不修改底座参数的前提下，显著提升多跳新答案成功率；
4. 在反事实场景中，注入向量需要结构化对齐，随机向量和弱化版本都明显更差；
5. 在时间更新场景中，CQAG 仍可能带来收益，但稳定性和局部性明显更弱，说明该任务更接近真实开放世界编辑难题。

## 4. 论文中可直接使用的实验表述建议

可在论文中将当前结果组织为以下结构：

1. **机制验证实验**：使用合成两跳环境，严格验证“单跳改对但多跳仍错”的推导断裂机理。
2. **真实模型实验**：在 `Qwen2.5-0.5B + MQuAKE-CF-3k-v2` 上验证 CQAG 对多跳反事实问题的提升。
3. **消融实验**：
   - `CQAG-full`
   - `CQAG-fact-only`
   - `CQAG-random`
   - `mismatched-vector locality`
4. **时间更新分析**：
   - 严格过滤下，当前公开中小模型难以形成稳定旧知识基线；
   - 在 `7B + generation filter` 条件下，CQAG 可取得初步提升，但稳定性不足。

## 5. 还可继续增强的方向

如果继续冲更完整的论文版本，最值得追加的是：

1. 将当前残差注入替换为更接近 Q-V 通道的 attention value hook；
2. 增加中文或中英混合问题，补跨语种实验；
3. 输出图表版结果，如柱状图、消融图和层-强度热图；
4. 若追求更强时间更新结论，可继续上探 `8B+` 或更高质量指令模型。
