import json
import statistics
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "paper_ready"

GROUPS = {
    "mquake_cf": [
        "value_cqag_cf_blind30_multilayer_s7.json",
        "value_cqag_cf_blind30_multilayer_s11.json",
        "value_cqag_cf_blind30_multilayer_s23.json",
    ],
    "mquake_t": [
        "value_cqag_t_blind16_s7.json",
        "value_cqag_t_blind16_s11.json",
        "value_cqag_t_blind16_s23.json",
    ],
}


def summarize(values: List[float]) -> Dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    output = {}
    for group, filenames in GROUPS.items():
        runs = []
        for filename in filenames:
            data = json.loads((RESULTS / filename).read_text(encoding="utf-8"))
            runs.append({"file": filename, "best": data["best"], "metrics": data["metrics"]})
        metric_names = list(runs[0]["metrics"])
        aggregate = {
            metric: summarize([float(run["metrics"][metric]) for run in runs])
            for metric in metric_names
            if metric not in {"dev_case_count", "test_case_count"}
        }
        output[group] = {
            "run_count": len(runs),
            "dev_case_count_per_run": runs[0]["metrics"]["dev_case_count"],
            "test_case_count_per_run": runs[0]["metrics"]["test_case_count"],
            "runs": runs,
            "aggregate": aggregate,
        }
    out_path = RESULTS / "value_cqag_multiseed_aggregate.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
