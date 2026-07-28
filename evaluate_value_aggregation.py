import argparse
import json
from pathlib import Path

import torch

from evaluate_value_router import jaccard, route
from run_real_cqag_experiment_strict import ExperimentConfig, matches, parse_int_list, set_seed
from run_value_cqag_question_specific import QuestionSpecificValueCQAG


MODES = ("nearest", "mean", "weighted")


def aggregate_vectors(question, remaining, vectors, layers, mode):
    if mode == "nearest":
        selected, _ = route(question, remaining, 0.0, content_only=True)
        return {layer: vectors[selected][layer] for layer in layers}
    weights = [1.0 for _ in remaining]
    if mode == "weighted":
        weights = [jaccard(question, source, content_only=True) for source in remaining]
        if sum(weights) == 0:
            weights = [1.0 for _ in remaining]
    total = sum(weights)
    return {
        layer: sum(
            (weight / total) * vectors[source][layer]
            for source, weight in zip(remaining, weights)
        )
        for layer in layers
    }


def evaluate(experiment, cases, layers, mode, alpha, token_window, threshold):
    details = []
    for case in cases:
        item = case["item"]
        source_questions = list(case["vectors"])
        for question in item["questions"]:
            remaining = [source for source in source_questions if source != question]
            selected, score = route(question, remaining, threshold, content_only=True)
            steered = selected is not None
            aggregated = (
                aggregate_vectors(question, remaining, case["vectors"], layers, mode)
                if steered
                else None
            )
            prediction = experiment.generate_value(
                question,
                aggregated,
                alpha if steered else 0.0,
                token_window,
            )
            success = matches(prediction, item["new_answer"], item.get("new_answer_alias", []))
            details.append(
                {
                    "case_id": int(item["case_id"]),
                    "question": question,
                    "routed": steered,
                    "route_score": score,
                    "success": success,
                    "prediction": prediction,
                }
            )
            print(f"[{mode}] case={item['case_id']} routed={steered} success={success}", flush=True)
    case_ids = sorted({row["case_id"] for row in details})
    return {
        "question_acc": sum(row["success"] for row in details) / len(details),
        "case_acc": sum(
            all(row["success"] for row in details if row["case_id"] == case_id)
            for case_id in case_ids
        )
        / len(case_ids),
        "route_recall": sum(row["routed"] for row in details) / len(details),
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare leave-one-out Value vector aggregation strategies.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--case-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="20,22,24,26")
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--token-window", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    case_ids = parse_int_list(args.case_ids)
    layers = tuple(parse_int_list(args.layers))
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
    cases = []
    for item in experiment.dataset:
        if int(item["case_id"]) not in wanted:
            continue
        cases.append(
            {"item": item, "vectors": experiment.build_question_vectors(item, list(layers))}
        )
        print(f"[vectors] case={item['case_id']}", flush=True)

    results = {
        mode: evaluate(
            experiment, cases, layers, mode, args.alpha, args.token_window, args.threshold
        )
        for mode in MODES
    }
    output = {
        "config": vars(args),
        "metrics": {
            mode: {key: value for key, value in result.items() if key != "details"}
            for mode, result in results.items()
        },
        "details": {mode: result["details"] for mode, result in results.items()},
    }
    temporary = Path(args.output + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(output["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
