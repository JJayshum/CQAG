# Frozen Blind Evaluation Report (2026-07-27)

## Protocol

- Dataset: MQuAKE-CF strict-qualified cases.
- Historical set: 40 cases used by earlier experiments.
- Development set: 20 newly collected cases, split into 10 calibration and 10 validation cases.
- Frozen blind test: 80 previously unused cases.
- Split seed: `20260727`.
- Blind-set SHA-256: `cc4060ec5b1068a372eeaccf1545c451f541c3395285fb5e7ec5e9953bcfc38b`.
- The blind set was evaluated once after configuration selection. No code or hyperparameter was tuned on its outcomes.
- Model: Qwen2.5-7B-Instruct, loaded in 4-bit mode.

The selected intervention constructs a separate no-leak Value-channel difference vector for every question. It uses Qwen attention `v_proj` layers 20, 22, 24, and 26, scale 5.0, and injects over the final 16 prompt tokens. The final multi-hop answer is not used to construct the vector.

## Development comparison

Both variants used the same 10-case calibration and 10-case held-out development validation split.

| Method | New-answer preference | Case generation | Question generation |
|---|---:|---:|---:|
| Case-average Value vector | 80.0% | 20.0% | 53.3% |
| Question-specific Value vector | 80.0% | 30.0% | 56.7% |

The question-specific four-layer configuration was frozen because it preserved preference accuracy and improved both free-generation measures on development validation.

## Frozen 80-case blind results

95% confidence intervals are Wilson score intervals. Case metrics use 80 observations and question generation uses 240 observations.

| Method | New-answer preference | Case generation | Question generation |
|---|---:|---:|---:|
| Qwen7B base | 0.00% [0.00, 4.58] | 2.50% [0.69, 8.66] | 3.75% [1.99, 6.97] |
| Explicit updated-fact prompt | 91.25% [83.02, 95.70] | 81.25% [71.34, 88.29] | 89.17% [84.60, 92.50] |
| Question-specific Value-CQAG | 76.25% [65.86, 84.24] | 23.75% [15.76, 34.14] | 40.83% [34.81, 47.15] |

Raw counts for Value-CQAG are 61/80 preference successes, 19/80 full-case generation successes, and 98/240 successful question generations.

## Interpretation

The frozen evaluation supports three claims:

1. The intervention has a strong, nontrivial effect. The base model never prefers the updated answer for every question in a case, while Value-CQAG succeeds on 61 of 80 cases.
2. The effect generalizes beyond the historical tuning cases. Blind preference accuracy remains close to the held-out development result (76.25% versus 80.0%).
3. Preference transfer is substantially stronger than exact free generation. Value-CQAG reaches 40.83% per-question generation, far above the 3.75% base rate, but well below the 89.17% explicit-prompt ceiling.

The development-to-blind decline in question generation (56.7% to 40.8%) shows why the frozen evaluation was necessary. The 20-case development set was too small for a reliable headline estimate.

## Publication readiness

These results are materially stronger and more defensible than the earlier 30-case evaluation, but they are not by themselves sufficient for a top-tier submission. A credible paper still needs:

- additional knowledge-editing baselines such as MEMIT under the same cases and metrics;
- unrelated-neighborhood locality and portability evaluations rather than only mismatched-direction controls;
- latency, memory, and throughput measurements;
- paired significance tests once all methods have per-case predictions;
- a clear framing of the method as inference-time activation steering, including its weaker free-generation performance and per-query vector-construction cost.

The present evidence supports a promising research result, not a claim that the project has already reached guaranteed top-conference acceptance quality.

## Cross-model replication: Llama-3.1-8B-Instruct

Directly reusing the Qwen late-layer search space produced weak development results. The Llama search was therefore expanded, on the same frozen 20-case development set only, to cover layers 8 through 28. The selected Llama configuration uses layers 16, 20, 24, and 28, scale 5.0, and a 16-token window. Its held-out development validation scores were 70.0% preference, 40.0% case generation, and 50.0% question generation.

The selected configuration was then run once on the same 80-case blind split.

| Llama-3.1-8B method | New-answer preference | Case generation | Question generation |
|---|---:|---:|---:|
| Base | 0.00% [0.00, 4.58] | 2.50% [0.69, 8.66] | 4.58% [2.58, 8.02] |
| Explicit updated-fact prompt | 92.50% [84.59, 96.52] | 83.75% [74.16, 90.25] | 92.50% [88.46, 95.20] |
| Question-specific Value-CQAG | 58.75% [47.80, 68.89] | 28.75% [19.99, 39.46] | 42.92% [36.81, 49.24] |

Raw Value-CQAG counts are 47/80 preference successes, 23/80 full-case generation successes, and 103/240 successful question generations. The replication confirms a substantial effect over the unprompted model, while also showing architecture sensitivity: Llama has lower preference transfer than Qwen but slightly higher exact generation.

## Standard knowledge-editing baseline: ROME

ROME was run through the official EasyEdit implementation and its supplied Qwen2.5-7B hyperparameters. Each MQuAKE case was edited independently, the model was evaluated with the same chat-template preference and generation metrics, and original weights were restored before the next case. Five questions from cases outside the frozen evaluation set were sampled per edit; locality requires the complete greedy-decoded output to remain exactly unchanged before and after editing.

| Qwen2.5-7B method | New-answer preference | Case generation | Question generation | Unrelated exact-output locality |
|---|---:|---:|---:|---:|
| ROME | 32.50% [23.24, 43.36] | 13.75% [7.85, 22.97] | 19.17% [14.69, 24.62] | 80.75% [76.60, 84.31] |
| Question-specific Value-CQAG | 76.25% [65.86, 84.24] | 23.75% [15.76, 34.14] | 40.83% [34.81, 47.15] | 3.33% [2.06, 5.35] |

ROME raw counts are 26/80 preference successes, 11/80 full-case generation successes, 46/240 successful question generations, and 323/400 unrelated outputs preserved exactly. Mean ROME editing time was 4.33 seconds per case, excluding evaluation. The result establishes that the Value-CQAG advantage is not only relative to an unedited base model; it also substantially exceeds a standard parametric editor on multi-hop transfer.

Official MEMIT configuration requires per-layer second-moment statistics computed from 100,000 Wikipedia samples. No compatible Qwen2.5-7B cache was available, so MEMIT was not reported with reduced or improvised statistics.

## Unrelated locality and efficiency audit

The earlier mismatched-direction control is not a sufficient locality measure. A stricter audit sampled two unrelated questions for each of the three question-specific vectors in every blind case. It compared complete greedy-decoded outputs with and without the intervention, matching the exact-output protocol used for ROME.

Value-CQAG preserved only 16 of 480 unrelated outputs exactly: 3.33% [2.06, 5.35]. ROME preserved 323 of 400: 80.75% [76.60, 84.31]. This is a major limitation and overturns any interpretation of the earlier mismatched-direction score as evidence of strong real-input locality.

Measured Qwen2.5-7B costs were:

| Operation | Mean time |
|---|---:|
| Build all question-specific vectors for one case | 2.27 s |
| Base generation per unrelated question | 0.388 s |
| Value-steered generation per unrelated question | 0.924 s |
| Steered/base generation latency ratio | 2.38x |

ROME required 4.33 seconds per case for weight editing, excluding evaluation. The methods therefore have different cost profiles: ROME pays a larger one-time edit cost and preserves unrelated outputs substantially better, while Value-CQAG avoids persistent weight mutation but pays per-case vector construction and per-query hook overhead.

## Paired significance

All tests use the same 80 blind cases. Exact McNemar tests and 20,000-sample paired bootstrap intervals give:

| Comparison | Metric difference | Paired bootstrap 95% CI | McNemar p |
|---|---:|---:|---:|
| Value-CQAG minus base | Preference +76.25 pp | [+66.25, +85.00] | 8.67e-19 |
| Value-CQAG minus ROME | Preference +43.75 pp | [+31.25, +56.25] | 2.84e-9 |
| Value-CQAG minus explicit prompt | Preference -15.00 pp | [-25.00, -5.00] | 0.00754 |
| Value-CQAG minus base | Case generation +21.25 pp | [+12.50, +31.25] | 7.63e-5 |
| Value-CQAG minus ROME | Case generation +10.00 pp | [0.00, +20.00] | 0.0963 |
| Value-CQAG minus explicit prompt | Case generation -57.50 pp | [-70.00, -45.00] | 1.04e-11 |

The preference advantage over ROME is decisive. The full-case generation advantage over ROME is not significant at the conventional 0.05 level, so it should be described as a numerical improvement rather than a confirmed one.
