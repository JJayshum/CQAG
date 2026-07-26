import json
import subprocess
from pathlib import Path


SEEDS = [1, 2, 3, 4, 5]
BASE_DIR = Path("/root/cqag_experiment/results/synth_multiseed")
BASE_DIR.mkdir(parents=True, exist_ok=True)


def mean(values):
    return sum(values) / len(values)


def main() -> None:
    per_seed = []
    for seed in SEEDS:
        out_dir = BASE_DIR / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "python",
            "/root/cqag_experiment/run_synthetic_cqag.py",
            "--seed",
            str(seed),
            "--output-dir",
            str(out_dir),
        ]
        subprocess.run(cmd, check=True)
        with open(out_dir / "summary.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        per_seed.append(
            {
                "seed": seed,
                "single_hop_edit_new": data["post_edit_no_cqag"]["single_hop_edit_new_target"]["accuracy"],
                "single_hop_locality": data["post_edit_no_cqag"]["single_hop_locality_old_target"]["accuracy"],
                "multi_hop_no_cqag_new": data["post_edit_no_cqag"]["multi_hop_edit_new_target"]["accuracy"],
                "multi_hop_no_cqag_old": data["post_edit_no_cqag"]["multi_hop_edit_old_target"]["accuracy"],
                "multi_hop_cqag_new": data["post_edit_cqag"]["multi_hop_edit_new_target"]["accuracy"],
                "multi_hop_cqag_old": data["post_edit_cqag"]["multi_hop_edit_old_target"]["accuracy"],
                "multi_hop_random": data["post_edit_cqag"]["multi_hop_random_vector"]["accuracy"],
                "alpha": data["post_edit_cqag"]["alpha"],
            }
        )

    agg = {
        "seed_count": len(per_seed),
        "single_hop_edit_new_mean": mean([x["single_hop_edit_new"] for x in per_seed]),
        "single_hop_locality_mean": mean([x["single_hop_locality"] for x in per_seed]),
        "multi_hop_no_cqag_new_mean": mean([x["multi_hop_no_cqag_new"] for x in per_seed]),
        "multi_hop_no_cqag_old_mean": mean([x["multi_hop_no_cqag_old"] for x in per_seed]),
        "multi_hop_cqag_new_mean": mean([x["multi_hop_cqag_new"] for x in per_seed]),
        "multi_hop_cqag_old_mean": mean([x["multi_hop_cqag_old"] for x in per_seed]),
        "multi_hop_random_mean": mean([x["multi_hop_random"] for x in per_seed]),
        "alpha_mean": mean([x["alpha"] for x in per_seed]),
        "per_seed": per_seed,
    }
    out_path = BASE_DIR / "aggregate.json"
    out_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
