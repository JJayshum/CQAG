import argparse
import json
import random
import time
from pathlib import Path

from run_real_cqag_experiment_strict import ExperimentConfig, parse_float_list, parse_int_list, set_seed
from run_value_cqag_experiment import parse_layer_sets
from run_value_cqag_question_specific import QuestionSpecificValueCQAG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--case-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="20,22,24,26")
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--token-window", type=int, default=16)
    parser.add_argument("--locality-questions-per-vector", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260727)
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
    unrelated_pool = [
        question
        for row in experiment.dataset
        if int(row["case_id"]) not in wanted
        for question in row.get("questions", [])
    ]
    rng = random.Random(args.seed)
    details = []

    for item in items:
        start = time.perf_counter()
        vectors = experiment.build_question_vectors(item, list(layers))
        vector_seconds = time.perf_counter() - start
        locality_hits = []
        base_seconds = 0.0
        steered_seconds = 0.0
        pairs = []
        for source_question in item["questions"]:
            sampled = rng.sample(
                unrelated_pool,
                min(args.locality_questions_per_vector, len(unrelated_pool)),
            )
            for unrelated_question in sampled:
                start = time.perf_counter()
                base_output = experiment.generate_value(unrelated_question, None, 0.0, args.token_window)
                base_seconds += time.perf_counter() - start
                start = time.perf_counter()
                steered_output = experiment.generate_value(
                    unrelated_question,
                    vectors[source_question],
                    args.alpha,
                    args.token_window,
                )
                steered_seconds += time.perf_counter() - start
                hit = base_output == steered_output
                locality_hits.append(hit)
                pairs.append(
                    {
                        "source_question": source_question,
                        "unrelated_question": unrelated_question,
                        "base_output": base_output,
                        "steered_output": steered_output,
                        "preserved": hit,
                    }
                )
        row = {
            "case_id": int(item["case_id"]),
            "vector_seconds": vector_seconds,
            "base_generation_seconds": base_seconds,
            "steered_generation_seconds": steered_seconds,
            "locality_hits": locality_hits,
            "pairs": pairs,
        }
        details.append(row)
        print(
            f"case={row['case_id']} locality={sum(locality_hits)}/{len(locality_hits)} "
            f"vector_s={vector_seconds:.3f} base_gen_s={base_seconds:.3f} "
            f"steered_gen_s={steered_seconds:.3f}",
            flush=True,
        )

    locality_hits = [hit for row in details for hit in row["locality_hits"]]
    pair_count = len(locality_hits)
    metrics = {
        "case_count": len(details),
        "locality_pair_count": pair_count,
        "exact_output_locality_acc": sum(locality_hits) / pair_count,
        "mean_vector_seconds_per_case": sum(row["vector_seconds"] for row in details) / len(details),
        "mean_base_generation_seconds_per_pair": sum(row["base_generation_seconds"] for row in details) / pair_count,
        "mean_steered_generation_seconds_per_pair": sum(row["steered_generation_seconds"] for row in details) / pair_count,
    }
    result = {"config": vars(args), "metrics": metrics, "details": details}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
