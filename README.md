# CQAG

CQAG（Coordinated Query-Value Activation Guiding）是一个面向知识编辑后多跳推理断裂问题的实验项目。

项目验证的核心现象是：模型在单跳问题中能够使用编辑后的新事实，但在多跳问题中仍可能沿用旧桥接事实。仓库包含合成机制实验、Qwen2.5 真实模型实验、MQuAKE-CF/MQuAKE-T 结果，以及对早期实验的严格审计。

## 最新可靠结果

模型：Qwen2.5-7B-Instruct（4bit）  
数据：MQuAKE-CF 40 条严格筛选案例

| 方法 | 新答案概率偏好 | 案例级自由生成 | 逐问题自由生成 |
|---|---:|---:|---:|
| 无辅助 | 0.0% | 0.0% | - |
| 显式更新事实提示 | 90.0% | 60.0% | - |
| 反事实世界约束 | 90.0% | **67.5%** | **76.7% (92/120)** |
| 隐式关系链追踪 | **92.5%** | 65.0% | - |

严格实验报告见 [STRICT_EXPERIMENT_UPDATE_2026-07-25.md](STRICT_EXPERIMENT_UPDATE_2026-07-25.md)。

## 重要实验审计说明

早期 `CQAG-full` 向量同时使用了编辑事实差分和最终多跳答案差分。后者直接包含测试样本的目标答案，存在答案信息泄漏。因此早期 full 指标仅保留为开发记录，不能作为无泄漏 CQAG 的最终结论。

去掉最终答案后，当前单层末 token 残差注入尚不能稳定复现显式事实提示的效果。现有结果支持以下更谨慎的结论：

1. 单跳编辑成功并不代表多跳推理会切换到新知识；
2. 更新事实本身足以显著修复多跳推理；
3. 当前主要瓶颈是如何把显式上下文可靠蒸馏为无泄漏的隐藏状态干预。

## 目录说明

- `run_synthetic_cqag.py`：合成两跳机制实验。
- `run_synthetic_multiseed.py`：合成实验多随机种子评测。
- `run_real_model_probe.py`：真实模型知识探测。
- `run_real_cqag_experiment.py`：早期真实模型实验，用于复现历史结果。
- `run_real_cqag_experiment_strict.py`：无答案泄漏、随机划分和生成评测版本。
- `eval_graph_prompt.py`：显式更新事实与反事实提示评测。
- `auto_paper_ready_experiments.py`：模型与配置扫描工具。
- `results/`：实验输出 JSON。
- `STRICT_EXPERIMENT_UPDATE_2026-07-25.md`：最新严格实验报告。

## 安装

```bash
pip install -r requirements.txt
```

真实模型实验需要支持 CUDA 的 PyTorch 环境。4bit 加载依赖 `bitsandbytes`。

## 数据准备

第三方 MQuAKE 数据未提交到仓库。请将数据放入：

```text
data/MQuAKE-CF-3k-v2.json
data/MQuAKE-T.json
```

脚本默认路径基于原实验服务器，可通过命令行参数覆盖模型、数据和输出路径。

## 快速运行

合成实验：

```bash
python run_synthetic_cqag.py
python run_synthetic_multiseed.py
```

严格真实模型实验示例：

```bash
python run_real_cqag_experiment_strict.py \
  --model-dir /path/to/Qwen2.5-7B-Instruct \
  --dataset-path data/MQuAKE-CF-3k-v2.json \
  --output results/paper_ready/strict_result.json \
  --load-in-4bit \
  --filter-mode strict
```

## 后续方向

- 多层、多 token 的 attention value/KV-cache 注入；
- 蒸馏显式关联事实提示的多层激活轨迹；
- 基于查询与编辑关联度的动态门控；
- 固定开发集与独立盲测集；
- 更完整的 locality、portability 和跨模型验证。
