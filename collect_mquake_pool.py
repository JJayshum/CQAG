import argparse
import json
from pathlib import Path

from run_real_cqag_experiment_strict import CQAGRealExperiment, ExperimentConfig, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect qualified MQuAKE cases with a real model.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--filter-mode", choices=["strict", "generation"], default="generation")
    parser.add_argument("--max-scan-cases", type=int, default=1800)
    parser.add_argument("--scan-start-case", type=int, default=0)
    parser.add_argument("--target-pool-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = ExperimentConfig(
        model_dir=args.model_dir,
        dataset_path=args.dataset_path,
        seed=args.seed,
        max_scan_cases=args.max_scan_cases,
        scan_start_case=args.scan_start_case,
        target_pool_size=args.target_pool_size,
        load_in_4bit=True,
        filter_mode=args.filter_mode,
    )
    set_seed(args.seed)
    experiment = CQAGRealExperiment(config)
    pool_info = experiment.collect_pool()
    result = {
        "config": vars(args),
        "qualified_case_ids": [int(item["case_id"]) for item in pool_info["pool"]],
        "qualified_pool_size": len(pool_info["pool"]),
        "probe_records": pool_info["probe_records"],
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "probe_records"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
