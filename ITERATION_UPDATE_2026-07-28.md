# CQAG Iteration Update (2026-07-28)

## Evaluation policy

- No final multi-hop answer is used to construct steering directions.
- Configuration selection and confirmatory evaluation use disjoint case ranges.
- Confirmatory comparisons are paired by case and use case-level bootstrap intervals.
- Failed hypotheses are retained below and are not promoted as improvements.

## Confirmatory pool

Qwen2.5-7B-Instruct (4-bit) was used to scan MQuAKE-CF cases after index 1073.
The strict filter requires correct old single-hop generation, correct old multi-hop generation,
and higher old-answer than new-answer likelihood. The first 100 qualifying cases form the
confirmatory pool. They are disjoint from all cases through 1072 used in prior development.

## Frozen confirmatory results (100 cases / 300 questions)

| Condition | Question generation | Case generation | Route recall |
|---|---:|---:|---:|
| Base model | 2.0% | 0.0% | n/a |
| Explicit updated facts | 95.0% | 90.0% | n/a |
| Value-CQAG, prompt prefill only | 38.67% | 18.0% | 93.67% |
| Value-CQAG, plus first 2 decode steps | 34.67% | not promoted | 93.67% |

Paired case-bootstrap differences:

- Prompt-only Value-CQAG minus base: +36.67 pp, 95% CI [+29.33, +44.33].
- Two decode steps minus prompt-only: -4.00 pp, 95% CI [-8.67, +0.33].

Conclusion: Value-CQAG has a large, reproducible effect over the base model, but extending
the intervention into decoding does not improve it and likely hurts. The 56.33 pp gap to the
explicit-prompt upper bound remains the main problem.

## Rejected development hypotheses

| Hypothesis | Development/validation result | Decision |
|---|---:|---|
| Token-aligned 16-token Value trajectory | 17.0% question / 5.0% case | Reject |
| Mean of two leave-one-out vectors | 38.0% question / 19.0% case | No meaningful gain |
| Similarity-weighted leave-one-out vectors | 38.67% question / 20.0% case | No meaningful gain |
| Joint Key+Value broadcast injection | 34.0% question / 22.0% case | Reject |

These experiments indicate that untrained linear activation differences at a fixed projection
layer have reached a practical ceiling. Additional alpha or layer scans on the same pools are
not justified.

## Remaining publication blockers

1. Free generation remains far below explicit prompting (38.67% versus 95.0%).
2. Existing exact-output locality without a router is poor; router locality and end-to-end
   generation locality must be reported separately.
3. The strict pool is relation-skewed, so micro accuracy must be accompanied by relation-macro
   results and confidence intervals.
4. Cross-model replication is incomplete; earlier Llama and Qwen-3B results are not yet a
   consistent confirmation of the same frozen method.
5. The method currently requires per-edit/per-query representation extraction and needs a fair
   end-to-end latency and memory comparison.

## Next method milestone

The next defensible iteration is a trained, low-capacity distillation module. It should learn on
development edits to reproduce the explicit-fact prompt's multi-layer attention behavior while
being constrained by unrelated-query locality. The final post-2064 strict pool must remain sealed
until the architecture, loss, hyperparameters, and stopping rule are frozen.

## Sealed final pool

The remaining MQuAKE-CF range after index 2064 was scanned with the same strict filter.
Sixty qualifying cases were saved to `results/paper_ready/mquake_cf_final_reserved_pool60_after2064.json`.
No steering method has been evaluated on these cases. Treat this file as sealed confirmatory data;
do not inspect per-case target outcomes or use it for architecture selection.
