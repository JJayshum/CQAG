import argparse
import json
import random
import time
from pathlib import Path

from evaluate_value_router import route
from run_real_cqag_experiment_strict import ExperimentConfig, matches, parse_int_list, set_seed
from run_value_cqag_question_specific import QuestionSpecificValueCQAG


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description="End-to-end locality for routed Value-CQAG.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--case-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="20,22,24,26")
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--token-window", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--unrelated-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260730)
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
    items = [row for row in experiment.dataset if int(row["case_id"]) in wanted]
    if len(items) != len(wanted):
        raise RuntimeError(f"Found {len(items)} of {len(wanted)} requested cases")

    unrelated = [
        {
            "case_id": int(row["case_id"]),
            "question": question,
            "answer": row["answer"],
        }
        for row in experiment.dataset
        if int(row["case_id"]) not in wanted
        for question in row.get("questions", [])
    ]
    rng = random.Random(args.seed)
    unrelated = rng.sample(unrelated, min(args.unrelated_count, len(unrelated)))

    base_predictions = {}
    for row in unrelated:
        base_predictions[row["question"]] = experiment.generate_value(
            row["question"], None, 0.0, args.token_window
        )

    details = []
    vector_seconds = 0.0
    injected_count = 0
    for case in items:
        item = case
        start = time.perf_counter()
        vectors = experiment.build_question_vectors(item, list(layers))
        vector_seconds += time.perf_counter() - start
        source_questions = list(vectors)
        for unrelated_row in unrelated:
            selected, score = route(
                unrelated_row["question"], source_questions, args.threshold, content_only=True
            )
            base_prediction = base_predictions[unrelated_row["question"]]
            if selected is None:
                routed_prediction = base_prediction
            else:
                injected_count += 1
                routed_prediction = experiment.generate_value(
                    unrelated_row["question"],
                    vectors[selected],
                    args.alpha,
                    args.token_window,
                )
            base_correct = routed_base_correct = False
            # The sampled question belongs to an untouched case; its original answer
            # is the correct target for the locality check.
            base_correct = matches(base_prediction, unrelated_row["answer"], [])
            routed_base_correct = matches(routed_prediction, unrelated_row["answer"], [])
            details.append(
                {
                    "edited_case_id": int(item["case_id"]),
                    "unrelated_case_id": unrelated_row["case_id"],
                    "question": unrelated_row["question"],
                    "selected_source_question": selected,
                    "route_score": score,
                    "route_hit": selected is not None,
                    "base_prediction": base_prediction,
                    "routed_prediction": routed_prediction,
                    "exact_output_preserved": base_prediction == routed_prediction,
                    "base_correct": base_correct,
                    "routed_correct": routed_base_correct,
                    "semantic_correctness_preserved": base_correct == routed_base_correct,
                }
            )
        print(f"[locality] case={item['case_id']} pairs={len(unrelated)}", flush=True)

    route_hits = [row["route_hit"] for row in details]
    exact = [row["exact_output_preserved"] for row in details]
    semantic = [row["semantic_correctness_preserved"] for row in details]
    false_route = [row for row in details if row["route_hit"]]
    metrics = {
        "edited_case_count": len(items),
        "unrelated_question_count": len(unrelated),
        "pair_count": len(details),
        "route_hit_count": sum(route_hits),
        "route_false_positive_rate": mean(route_hits),
        "router_rejection_rate": 1.0 - mean(route_hits),
        "end_to_end_exact_output_locality": mean(exact),
        "end_to_end_semantic_locality": mean(semantic),
        "exact_locality_conditional_on_route_hit": mean(
            row["exact_output_preserved"] for row in false_route
        ) if false_route else 1.0,
        "semantic_locality_conditional_on_route_hit": mean(
            row["semantic_correctness_preserved"] for row in false_route
        ) if false_route else 1.0,
        "base_accuracy_on_unrelated_questions": mean(row["base_correct"] for row in details),
        "routed_accuracy_on_unrelated_questions": mean(row["routed_correct"] for row in details),
        "newly_corrupted_unrelated_outputs": sum(
            row["base_correct"] and not row["routed_correct"] for row in details
        ),
        "newly_fixed_unrelated_outputs": sum(
            not row["base_correct"] and row["routed_correct"] for row in details
        ),
        "injected_pair_count": injected_count,
        "mean_vector_seconds_per_case": vector_seconds / len(items),
    }
    result = {
        "config": vars(args),
        "metrics": metrics,
        "unrelated_questions": unrelated,
        "details": details,
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
