#!/usr/bin/env bash
set -euo pipefail

cd /root/cqag_experiment

# Reuse the frozen confirmation IDs from the recorded Llama baseline.
case_ids="$(/root/miniconda3/bin/python -c "import json; print(json.load(open('results/paper_ready/llama31_8b_confirmatory100_explicit.json'))['config']['case_ids'])")"

exec /root/miniconda3/bin/python run_easyedit_mquake_baseline.py \
  --easyedit-dir /root/cqag_experiment/third_party/EasyEdit \
  --hparams /root/cqag_experiment/easyedit_hparams/MEMIT_llama3.1_8b_cqag.yaml \
  --dataset-path /root/cqag_experiment/data/MQuAKE-CF-3k-v2.json \
  --case-ids "$case_ids" \
  --output /root/cqag_experiment/results/paper_ready/memit_llama31_8b_confirmatory100.json \
  --algorithm MEMIT \
  --max-new-tokens 24 \
  --locality-questions 5 \
  --seed 20260732
