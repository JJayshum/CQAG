import argparse
import json
from pathlib import Path

from evaluate_value_router import route
from run_real_cqag_experiment_strict import ExperimentConfig, matches, parse_float_list, parse_int_list, set_seed
from run_value_cqag_question_specific import QuestionSpecificValueCQAG


def evaluate(experiment, cases, alpha, generation_steps, threshold, token_window):
    details = []
    for case in cases:
        item = case["item"]
        source_questions = list(case["vectors"])
        for question in item["questions"]:
            remaining = [source for source in source_questions if source != question]
            selected, score = route(question, remaining, threshold, content_only=True)
            if selected is None:
                prediction = experiment.generate_value(question, None, 0.0, token_window)
            else:
                prediction = experiment.generate_value(
                    question,
                    case["vectors"][selected],
                    alpha,
                    token_window,
                    generation_steps=generation_steps,
                )
            details.append(
                {
                    "case_id": int(item["case_id"]),
                    "question": question,
                    "routed": selected is not None,
                    "route_score": score,
                    "generation_success": matches(
                        prediction, item["new_answer"], item.get("new_answer_alias", [])
                    ),
                    "prediction": prediction,
                }
            )
    case_ids = sorted({row["case_id"] for row in details})
    case_successes = [
        all(row["generation_success"] for row in details if row["case_id"] == case_id)
        for case_id in case_ids
    ]
    return {
        "question_acc": sum(row["generation_success"] for row in details) / len(details),
        "case_acc": sum(case_successes) / len(case_successes),
        "route_recall": sum(row["routed"] for row in details) / len(details),
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--calibration-case-ids", required=True)
    parser.add_argument("--validation-case-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="20,22,24,26")
    parser.add_argument("--alpha-grid", default="2.5,5,7.5")
    parser.add_argument("--generation-step-grid", default="0,1,2,4")
    parser.add_argument("--token-window", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    calibration_ids = parse_int_list(args.calibration_case_ids)
    validation_ids = parse_int_list(args.validation_case_ids)
    all_ids = calibration_ids + validation_ids
    layers = list(parse_int_list(args.layers))
    config = ExperimentConfig(
        model_dir=args.model_dir,
        dataset_path=args.dataset_path,
        seed=args.seed,
        load_in_4bit=True,
        qualified_case_ids=all_ids,
    )
    set_seed(args.seed)
    experiment = QuestionSpecificValueCQAG(config)
    wanted = set(all_ids)
    cases = []
    for item in experiment.dataset:
        if int(item["case_id"]) in wanted:
            cases.append({"item": item, "vectors": experiment.build_question_vectors(item, layers)})
    by_id = {int(case["item"]["case_id"]): case for case in cases}
    calibration = [by_id[case_id] for case_id in calibration_ids]
    validation = [by_id[case_id] for case_id in validation_ids]

    grid = []
    for alpha in parse_float_list(args.alpha_grid):
        for generation_steps in parse_int_list(args.generation_step_grid):
            metrics = evaluate(
                experiment,
                calibration,
                alpha,
                generation_steps,
                args.threshold,
                args.token_window,
            )
            row = {
                "alpha": alpha,
                "generation_steps": generation_steps,
                "question_acc": metrics["question_acc"],
                "case_acc": metrics["case_acc"],
                "route_recall": metrics["route_recall"],
            }
            grid.append(row)
            print(f"[calibration] {row}", flush=True)
    best = max(grid, key=lambda row: (row["question_acc"], row["case_acc"], -row["generation_steps"]))
    validation_metrics = evaluate(
        experiment,
        validation,
        best["alpha"],
        best["generation_steps"],
        args.threshold,
        args.token_window,
    )
    result = {
        "config": vars(args),
        "calibration_grid": grid,
        "best": best,
        "validation": validation_metrics,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "validation": {k: v for k, v in validation_metrics.items() if k != "details"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
