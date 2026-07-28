# CQAG 投稿前补实验报告

更新日期：2026-07-28

## 结论摘要

本轮在原有 Qwen2.5-7B 严格确认性评测基础上，补充了第二强模型、显式事实与检索式上下文基线、路由后的端到端 locality，以及逐问题错误审计。所有新增主结果均使用既有的 100-case / 300-question MQuAKE-CF 确认池（case IDs 1074--2064），没有打开最后封存的 60-case 保留池。

最重要的结论有三点：

1. 无泄漏 Value-CQAG 在 Qwen2.5-7B 与 Llama-3.1-8B 上都显著优于未干预模型，但仍明显落后于显式更新事实和检索式上下文。
2. 路由使 Qwen 的完整系统 exact-output locality 达到 98.49%，semantic locality 达到 99.62%；但误路由一旦发生，底层注入本身几乎总会改变输出。
3. 当前最可信的论文不是“超过所有知识编辑器的新 SOTA”，而是关于多跳编辑失效、泄漏审计、激活干预跨模型可复现性以及效能--locality 权衡的严谨实证研究。

## 同池跨模型主结果

| 模型 | 方法 | 逐问题生成 | 95% CI | 案例级生成 | 与 Base 的配对差值 95% CI |
|---|---|---:|---:|---:|---:|
| Qwen2.5-7B | Base | 2.00% | 0.33--4.33% | 0.0% | -- |
| Qwen2.5-7B | Routed Value-CQAG | 38.67% | 31.33--46.33% | 18.0% | +36.67 pp [29.33, 44.33] |
| Qwen2.5-7B | Explicit facts | 95.00% | -- | 90.0% | -- |
| Llama-3.1-8B | Base | 4.00% | 1.33--7.00% | 0.0% | -- |
| Llama-3.1-8B | Routed Value-CQAG | 51.33% | 43.67--59.33% | 32.0% | +47.33 pp [39.33, 55.33] |
| Llama-3.1-8B | Explicit facts | 93.00% | -- | 87.0% | -- |

两模型都显示持续到前两个解码步的注入会降低效果：Qwen 为 -4.00 pp，Llama 为 -7.00 pp（95% CI [-11.00, -3.33]）。因此论文应使用 prompt-only 注入作为最终方法，而不应再把持续解码注入描述为候选优势。

## IKE-style 检索基线

在相同 Qwen 确认池上，使用 500 个不重叠开发案例构造检索语料、MiniLM 语义检索和 8 个 ICL demonstrations 的透明 IKE-style 基线得到：

| 方法 | 逐问题生成 | 案例级生成 | 新答案偏好 | Exact-output locality | Semantic locality |
|---|---:|---:|---:|---:|---:|
| IKE-style retrieval prompt | 95.0% | 93.0% | 99.0% | 25.0% | 61.4% |
| Routed Value-CQAG | 38.67% | 18.0% | -- | 98.49% | 99.62% |

该基线使用当前编辑的更新事实并检索其他编辑示例，属于透明的 retrieval-plus-ICL 对照，而非官方 EasyEdit IKE 的逐字复现。它证明显式上下文足以解决绝大多数问题，同时也显示无门控检索上下文会显著损害无关输出的稳定性。

## 路由后的端到端 locality

Qwen2.5-7B 上固定使用 content-token Jaccard router、阈值 0.35，在 100 个编辑案例和 100 个无关问题形成的 10,000 对中：

| 指标 | 结果 |
|---|---:|
| Router false-positive rate | 1.52% (152 / 10,000) |
| End-to-end exact-output locality | 98.49% |
| End-to-end semantic locality | 99.62% |
| Exact locality given false route | 0.66% |
| Semantic locality given false route | 75.0% |
| Newly corrupted originally-correct outputs | 37 |
| Newly fixed originally-incorrect outputs | 1 |

因此系统层面可以主张“高 routed locality”，但不能声称注入本身 local。误路由是当前方法最清晰的安全边界和后续工作方向。

## 错误类型

Qwen Value-CQAG 的 300 个问题中，116 个正确生成新答案（38.67%）；133 个为其他实体或推理错误（44.33%），31 个仍输出旧答案（10.33%），19 个由 router miss 导致（6.33%），拒答仅 1 个。第三个问题改写的正确数最低（33 / 100），说明自由生成的主要瓶颈不是拒答，也不完全是旧知识残留，而是多跳组合后的错误实体选择或推理失败。

Llama 的对应错误构成是：154 个正确（51.33%）、96 个其他实体或推理错误（32.0%）、30 个旧答案残留（10.0%）、19 个 router miss（6.33%）和 1 个拒答。两模型的 router miss 数完全一致，表明当前瓶颈可拆分为路由覆盖和被路由后的生成可靠性两部分。

## MEMIT 状态

官方 EasyEdit MEMIT 对 Qwen2.5-7B 的配置需要对 Wikipedia 100,000 个样本计算每层二阶矩。服务器没有兼容缓存。为避免把缩小统计量或不同语料的近似结果误报为 MEMIT，本轮没有报告 MEMIT 数值。已有官方 ROME / EasyEdit 80-case 结果可保留为历史补充，但不应与本 100-case 主表混合比较。

提交长文前需要完成下列之一：

1. 按官方 100k Wikipedia 设置完成 MEMIT 统计并在冻结协议下评测；或
2. 把论文定位明确收缩为 activation steering 与 retrieval baseline 的实证分析，不再主张对完整参数编辑基线的全面比较。

## 投稿判断

当前证据足以支持 EACL 2027 short paper 的立即投稿，也足以在完成官方 MEMIT 后作为 NAACL 2027 / COLING 2027 长文候选。它仍不适合宣称顶会主会 SOTA 方法论文：检索式基线接近显式上限，而 Value-CQAG 的优势集中在 routed locality 和不修改模型权重，而非最终任务准确率。
