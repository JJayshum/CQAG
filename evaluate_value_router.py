import argparse
import json
import random
import re
import time
from pathlib import Path

from run_real_cqag_experiment_strict import (
    ExperimentConfig,
    matches,
    parse_float_list,
    parse_int_list,
    set_seed,
)
from run_value_cqag_question_specific import QuestionSpecificValueCQAG


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for",
    "from", "had", "has", "have", "he", "her", "his", "how", "in", "is", "it", "its",
    "of", "on", "or", "she", "that", "the", "their", "them", "they", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "whom", "whose", "with",
}


def tokens(text: str, content_only: bool = False) -> set[str]:
    result = set(re.findall(r"[a-z0-9]+", text.lower()))
    return result - STOPWORDS if content_only else result


def jaccard(a: str, b: str, content_only: bool = False) -> float:
    left, right = tokens(a, content_only), tokens(b, content_only)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def route(
    query: str, source_questions: list[str], threshold: float, content_only: bool
) -> tuple[str | None, float]:
    scored = [(question, jaccard(query, question, content_only)) for question in source_questions]
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
    parser.add_argument("--similarity-mode", choices=["lexical", "content"], default="lexical")
    parser.add_argument("--evaluate-leave-one-out", action="store_true")
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
    content_only = args.similarity_mode == "content"
    sampled_unrelated = rng.sample(unrelated_pool, min(args.locality_questions, len(unrelated_pool)))
    cases = []
    for item in items:
        start = time.perf_counter()
        vectors = experiment.build_question_vectors(item, list(layers))
        cases.append({"item": item, "vectors": vectors, "vector_seconds": time.perf_counter() - start})

    grid = []
    for threshold in thresholds:
        target_routes = []
        leave_one_out_routes = []
        leave_one_out_scores = []
        unrelated_routes = []
        unrelated_scores = []
        for case in cases:
            source_questions = list(case["vectors"])
            for question in case["item"]["questions"]:
                selected, _ = route(question, source_questions, threshold, content_only)
                target_routes.append(selected is not None)
                remaining = [source for source in source_questions if source != question]
                selected_loo, score_loo = route(question, remaining, threshold, content_only)
                leave_one_out_routes.append(selected_loo is not None)
                leave_one_out_scores.append(score_loo)
            for question in sampled_unrelated:
                selected, score = route(question, source_questions, threshold, content_only)
                unrelated_routes.append(selected is not None)
                unrelated_scores.append(score)
        row = {
            "threshold": threshold,
            "target_route_recall": sum(target_routes) / len(target_routes),
            "leave_one_out_paraphrase_route_recall": sum(leave_one_out_routes) / len(leave_one_out_routes),
            "mean_leave_one_out_score": sum(leave_one_out_scores) / len(leave_one_out_scores),
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
    if args.evaluate_leave_one_out:
        threshold = float(best["threshold"])
        portability = []
        for case in cases:
            item = case["item"]
            source_questions = list(case["vectors"])
            for question in item["questions"]:
                remaining = [source for source in source_questions if source != question]
                selected, score = route(question, remaining, threshold, content_only)
                if selected is None:
                    preference = False
                    prediction = experiment.generate_value(question, None, 0.0, args.token_window)
                else:
                    vector = case["vectors"][selected]
                    old_score = experiment.answer_logprob_value(
                        question, item["answer"], vector, args.alpha, args.token_window
                    )
                    new_score = experiment.answer_logprob_value(
                        question, item["new_answer"], vector, args.alpha, args.token_window
                    )
                    preference = new_score > old_score
                    prediction = experiment.generate_value(
                        question, vector, args.alpha, args.token_window
                    )
                generation = matches(
                    prediction, item["new_answer"], item.get("new_answer_alias", [])
                )
                portability.append(
                    {
                        "case_id": int(item["case_id"]),
                        "question": question,
                        "selected_source_question": selected,
                        "route_score": score,
                        "preference_success": preference,
                        "generation_success": generation,
                        "prediction": prediction,
                    }
                )
                print(
                    f"[portability] case={item['case_id']} routed={selected is not None} "
                    f"pref={preference} gen={generation}",
                    flush=True,
                )
        result["leave_one_out_portability"] = {
            "question_count": len(portability),
            "route_recall": sum(row["selected_source_question"] is not None for row in portability)
            / len(portability),
            "preference_acc": sum(row["preference_success"] for row in portability) / len(portability),
            "generation_acc": sum(row["generation_success"] for row in portability) / len(portability),
            "details": portability,
        }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
