import argparse
import json
import random
from pathlib import Path

from evaluate_value_router import route
from run_real_cqag_experiment_strict import ExperimentConfig, matches, parse_int_list, set_seed
from run_value_cqag_question_specific import QuestionSpecificValueCQAG


CONDITIONS = {
    "base": {"alpha": 0.0, "generation_steps": 0},
    "prompt_only": {"alpha": 5.0, "generation_steps": 0},
    "decode_steps_2": {"alpha": 5.0, "generation_steps": 2},
}


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def bootstrap_case_interval(rows, value_key, seed, samples=10000):
    by_case = {}
    for row in rows:
        by_case.setdefault(row["case_id"], []).append(float(row[value_key]))
    case_values = [mean(values) for values in by_case.values()]
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        draws.append(mean(rng.choice(case_values) for _ in case_values))
    draws.sort()
    return [draws[int(0.025 * samples)], draws[int(0.975 * samples)]]


def paired_bootstrap_interval(left_rows, right_rows, value_key, seed, samples=10000):
    left = {(row["case_id"], row["question"]): float(row[value_key]) for row in left_rows}
    right = {(row["case_id"], row["question"]): float(row[value_key]) for row in right_rows}
    if left.keys() != right.keys():
        raise ValueError("Paired conditions do not contain the same questions")
    by_case = {}
    for key in left:
        by_case.setdefault(key[0], []).append(right[key] - left[key])
    case_deltas = [mean(values) for values in by_case.values()]
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        draws.append(mean(rng.choice(case_deltas) for _ in case_deltas))
    draws.sort()
    return {
        "delta": mean(case_deltas),
        "ci95": [draws[int(0.025 * samples)], draws[int(0.975 * samples)]],
    }


def write_json_atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def evaluate_condition(experiment, cases, condition, layers, token_window, threshold):
    rows = []
    alpha = condition["alpha"]
    generation_steps = condition["generation_steps"]
    for case in cases:
        item = case["item"]
        source_questions = list(case["vectors"])
        for question in item["questions"]:
            selected = None
            route_score = 0.0
            vectors = None
            if alpha:
                remaining = [source for source in source_questions if source != question]
                selected, route_score = route(question, remaining, threshold, content_only=True)
                if selected is not None:
                    vectors = {
                        layer: case["vectors"][selected][layer]
                        for layer in layers
                    }
            prediction = experiment.generate_value(
                question,
                vectors,
                alpha if vectors is not None else 0.0,
                token_window,
                generation_steps=generation_steps,
            )
            success = matches(prediction, item["new_answer"], item.get("new_answer_alias", []))
            row = {
                "case_id": int(item["case_id"]),
                "question": question,
                "selected_source_question": selected,
                "route_score": route_score,
                "success": success,
                "prediction": prediction,
            }
            rows.append(row)
            print(
                f"[{condition['name']}] case={item['case_id']} routed={selected is not None} "
                f"success={success}",
                flush=True,
            )
    case_ids = sorted({row["case_id"] for row in rows})
    case_hits = [
        all(row["success"] for row in rows if row["case_id"] == case_id)
        for case_id in case_ids
    ]
    return {
        "question_acc": mean(row["success"] for row in rows),
        "question_acc_ci95": bootstrap_case_interval(rows, "success", seed=20260729),
        "case_acc": mean(case_hits),
        "route_recall": mean(row["selected_source_question"] is not None for row in rows)
        if alpha
        else None,
        "details": rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Frozen, paired blind evaluation for Value-CQAG generation schedules.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--case-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="20,22,24,26")
    parser.add_argument("--token-window", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--resume", action="store_true")
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
    items = [item for item in experiment.dataset if int(item["case_id"]) in wanted]
    if len(items) != len(wanted):
        raise RuntimeError(f"Found {len(items)} of {len(wanted)} requested cases")

    cases = []
    for item in items:
        cases.append(
            {
                "item": item,
                "vectors": experiment.build_question_vectors(item, list(layers)),
            }
        )
        print(f"[vectors] case={item['case_id']}", flush=True)

    partial_path = Path(args.output + ".partial.json")
    results = {}
    if args.resume and partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        results = partial.get("results", {})
        print(f"[resume] completed_conditions={list(results)}", flush=True)
    for name, values in CONDITIONS.items():
        if name in results:
            continue
        condition = {"name": name, **values}
        results[name] = evaluate_condition(
            experiment,
            cases,
            condition,
            layers,
            args.token_window,
            args.threshold,
        )
        write_json_atomic(
            partial_path,
            {
                "protocol": {
                    "case_ids": list(case_ids),
                    "conditions": CONDITIONS,
                    "layers": list(layers),
                    "token_window": args.token_window,
                    "threshold": args.threshold,
                    "seed": args.seed,
                },
                "results": results,
            },
        )
        print(f"[checkpoint] condition={name} path={partial_path}", flush=True)

    comparisons = {
        "prompt_only_minus_base": paired_bootstrap_interval(
            results["base"]["details"], results["prompt_only"]["details"], "success", args.seed
        ),
        "decode_steps_2_minus_prompt_only": paired_bootstrap_interval(
            results["prompt_only"]["details"],
            results["decode_steps_2"]["details"],
            "success",
            args.seed + 1,
        ),
    }
    output = {
        "protocol": {
            "confirmatory_primary_comparison": "decode_steps_2_minus_prompt_only",
            "case_ids": list(case_ids),
            "conditions": CONDITIONS,
            "layers": list(layers),
            "token_window": args.token_window,
            "threshold": args.threshold,
            "seed": args.seed,
        },
        "metrics": {
            name: {key: value for key, value in result.items() if key != "details"}
            for name, result in results.items()
        },
        "comparisons": comparisons,
        "details": {name: result["details"] for name, result in results.items()},
    }
    write_json_atomic(args.output, output)
    print(json.dumps({"metrics": output["metrics"], "comparisons": comparisons}, indent=2), flush=True)


if __name__ == "__main__":
    main()
