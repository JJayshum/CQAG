import argparse
import json
import math
import random
from pathlib import Path


def exact_mcnemar(a: list[bool], b: list[bool]) -> dict:
    b_only = sum((not x) and y for x, y in zip(a, b))
    a_only = sum(x and (not y) for x, y in zip(a, b))
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(0, min(a_only, b_only) + 1)) / (2**n)
        p = min(1.0, 2 * tail)
    return {"a_only": a_only, "b_only": b_only, "discordant": n, "exact_p": p}


def bootstrap_difference(a: list[bool], b: list[bool], seed: int, samples: int) -> dict:
    rng = random.Random(seed)
    n = len(a)
    values = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        values.append(sum(int(a[i]) - int(b[i]) for i in indices) / n)
    values.sort()
    return {
        "difference": sum(map(int, a)) / n - sum(map(int, b)) / n,
        "ci95": [values[int(0.025 * samples)], values[int(0.975 * samples)]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--value", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--rome", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    value = json.loads(Path(args.value).read_text())
    baselines = json.loads(Path(args.baselines).read_text())
    rome = json.loads(Path(args.rome).read_text())
    value_by_id = {int(row["case_id"]): row for row in value["details"]}
    base_by_id = {int(row["case_id"]): row for row in baselines["details"]}
    rome_by_id = {int(row["case_id"]): row for row in rome["details"]}
    ids = sorted(set(value_by_id) & set(base_by_id) & set(rome_by_id))

    methods = {
        "value": {
            "preference": [value_by_id[i]["preference_success"] for i in ids],
            "case_generation": [value_by_id[i]["generation_case_success"] for i in ids],
        },
        "base": {
            "preference": [base_by_id[i]["base_preference"] for i in ids],
            "case_generation": [base_by_id[i]["base_generation"] for i in ids],
        },
        "prompt": {
            "preference": [base_by_id[i]["prompt_preference"] for i in ids],
            "case_generation": [base_by_id[i]["prompt_generation"] for i in ids],
        },
        "rome": {
            "preference": [rome_by_id[i]["preference_success"] for i in ids],
            "case_generation": [rome_by_id[i]["generation_case_success"] for i in ids],
        },
    }
    result = {"case_ids": ids, "comparisons": {}}
    for metric in ("preference", "case_generation"):
        for other in ("base", "rome", "prompt"):
            key = f"value_vs_{other}_{metric}"
            a, b = methods["value"][metric], methods[other][metric]
            result["comparisons"][key] = {
                "mcnemar": exact_mcnemar(a, b),
                "paired_bootstrap": bootstrap_difference(a, b, args.seed, args.samples),
            }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
