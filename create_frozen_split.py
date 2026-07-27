import argparse
import hashlib
import json
import random
from pathlib import Path


def digest(ids: list[int]) -> str:
    payload = ",".join(map(str, ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--historical-count", type=int, default=40)
    parser.add_argument("--dev-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    pool = json.loads(Path(args.pool).read_text(encoding="utf-8"))["qualified_case_ids"]
    historical = pool[: args.historical_count]
    independent = pool[args.historical_count :]
    random.Random(args.seed).shuffle(independent)
    development = independent[: args.dev_count]
    blind_test = independent[args.dev_count :]
    manifest = {
        "seed": args.seed,
        "source_pool": args.pool,
        "historical_case_ids": historical,
        "development_case_ids": development,
        "blind_test_case_ids": blind_test,
        "counts": {
            "historical": len(historical),
            "development": len(development),
            "blind_test": len(blind_test),
        },
        "sha256": {
            "historical": digest(historical),
            "development": digest(development),
            "blind_test": digest(blind_test),
        },
        "policy": "Do not tune code or hyperparameters on blind_test_case_ids.",
    }
    Path(args.output).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
