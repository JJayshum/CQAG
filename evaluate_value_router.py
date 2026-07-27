import argparse
import json
import random
import re
import time
from pathlib import Path

from run_real_cqag_experiment_strict import ExperimentConfig, parse_float_list, parse_int_list, set_seed
from run_value_cqag_question_specific import QuestionSpecificValueCQAG


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    left, right = tokens(a), tokens(b)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def route(query: str, source_questions: list[str], threshold: float) -> tuple[str | None, float]:
    scored = [(question, jaccard(query, question)) for question in source_questions]
    question, score = max(scored, key=lambda row: row[1])
    return (question if score >= threshold else None), score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--case-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="20,22,24,26")
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--token-window", type=int, default=16)
    parser.add_argument("--threshold-grid", default="0.1,0.2,0.3,0.4,0.5,0.6")
    parser.add_argument("--locality-questions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    case_ids = parse_int_list(args.case_ids)
    layers = tuple(parse_int_list(args.layers))
    thresholds = parse_float_list(args.threshold_grid)
    config = ExperimentConfig(
        model_dir=args.model_dir,
        dataset_path=args.dataset_path,
        seed=args.seed,
        load_in_4bit=True,
        qualified_case_ids=case_ids,
    )
    set_seed(args.seed)
    experiment = QuestionSpecificValueCQAG(config)
    wanted = set(case_ids)
    items = [row for row in experiment.dataset if int(row["case_id"]) in wanted]
    unrelated_pool = [
        question
        for row in experiment.dataset
        if int(row["case_id"]) not in wanted
        for question in row.get("questions", [])
    ]
    rng = random.Random(args.seed)
    sampled_unrelated = rng.sample(unrelated_pool, min(args.locality_questions, len(unrelated_pool)))
    cases = []
    for item in items:
        start = time.perf_counter()
        vectors = experiment.build_question_vectors(item, list(layers))
        cases.append({"item": item, "vectors": vectors, "vector_seconds": time.perf_counter() - start})

    grid = []
    for threshold in thresholds:
        target_routes = []
        unrelated_routes = []
        unrelated_scores = []
        for case in cases:
            source_questions = list(case["vectors"])
            for question in case["item"]["questions"]:
                selected, _ = route(question, source_questions, threshold)
                target_routes.append(selected is not None)
            for question in sampled_unrelated:
                selected, score = route(question, source_questions, threshold)
                unrelated_routes.append(selected is not None)
                unrelated_scores.append(score)
        row = {
            "threshold": threshold,
            "target_route_recall": sum(target_routes) / len(target_routes),
            "unrelated_route_rate": sum(unrelated_routes) / len(unrelated_routes),
            "router_locality": 1.0 - sum(unrelated_routes) / len(unrelated_routes),
            "max_unrelated_score": max(unrelated_scores),
        }
        grid.append(row)
        print(row, flush=True)

    valid = [row for row in grid if row["target_route_recall"] == 1.0]
    best = max(valid, key=lambda row: (row["router_locality"], row["threshold"]))
    result = {
        "config": vars(args),
        "grid": grid,
        "best": best,
        "mean_vector_seconds_per_case": sum(row["vector_seconds"] for row in cases) / len(cases),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
