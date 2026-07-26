import argparse
import json
import time
from pathlib import Path

from run_real_cqag_experiment_strict import CQAGRealExperiment, ExperimentConfig, matches, set_seed


def augmented_prompt(experiment: CQAGRealExperiment, item: dict, question: str) -> str:
    facts = " ".join(
        f"{rr['prompt'].format(rr['subject'])} {rr['target_new']['str']}."
        for rr in item["requested_rewrite"]
    )
    return (
        "You are reasoning inside a hypothetical updated world. Treat the updated facts below as authoritative, "
        "even if they conflict with prior knowledge. Combine them with unchanged background knowledge. "
        "Do not discuss contradictions.\n"
        f"Updated facts: {facts}\nQuestion: {question}\nAnswer with just the final answer."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--case-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    case_ids = [int(x) for x in args.case_ids.split(",") if x.strip()]
    config = ExperimentConfig(model_dir=args.model_dir, dataset_path=args.dataset_path, seed=args.seed, load_in_4bit=True)
    set_seed(args.seed)
    experiment = CQAGRealExperiment(config)
    wanted = set(case_ids)
    items = [item for item in experiment.dataset if int(item["case_id"]) in wanted]
    details = []
    elapsed = {"base_logprob": 0.0, "prompt_logprob": 0.0, "base_generation": 0.0, "prompt_generation": 0.0}

    for item in items:
        base_pref, prompt_pref, base_gen, prompt_gen = [], [], [], []
        for question in item["questions"]:
            start = time.perf_counter()
            old_base = experiment.answer_logprob(question, item["answer"])
            new_base = experiment.answer_logprob(question, item["new_answer"])
            elapsed["base_logprob"] += time.perf_counter() - start
            base_pref.append(new_base > old_base)

            prompted = augmented_prompt(experiment, item, question)
            start = time.perf_counter()
            old_prompt = experiment.answer_logprob(prompted, item["answer"])
            new_prompt = experiment.answer_logprob(prompted, item["new_answer"])
            elapsed["prompt_logprob"] += time.perf_counter() - start
            prompt_pref.append(new_prompt > old_prompt)

            start = time.perf_counter()
            base_pred = experiment.generate_answer(question)
            elapsed["base_generation"] += time.perf_counter() - start
            base_gen.append(matches(base_pred, item["new_answer"], item.get("new_answer_alias", [])))

            start = time.perf_counter()
            prompt_pred = experiment.generate_answer(prompted)
            elapsed["prompt_generation"] += time.perf_counter() - start
            prompt_gen.append(matches(prompt_pred, item["new_answer"], item.get("new_answer_alias", [])))
        details.append(
            {
                "case_id": int(item["case_id"]),
                "base_preference": all(base_pref),
                "prompt_preference": all(prompt_pref),
                "base_generation": all(base_gen),
                "prompt_generation": all(prompt_gen),
                "base_question_hits": base_gen,
                "prompt_question_hits": prompt_gen,
            }
        )
        print(
            f"case={item['case_id']} base_pref={details[-1]['base_preference']} "
            f"prompt_pref={details[-1]['prompt_preference']} base_gen={details[-1]['base_generation']} "
            f"prompt_gen={details[-1]['prompt_generation']}",
            flush=True,
        )

    question_base = [x for row in details for x in row["base_question_hits"]]
    question_prompt = [x for row in details for x in row["prompt_question_hits"]]
    metrics = {
        "test_case_count": len(details),
        "base_new_preference_acc": sum(int(x["base_preference"]) for x in details) / len(details),
        "prompt_new_preference_acc": sum(int(x["prompt_preference"]) for x in details) / len(details),
        "base_generation_case_acc": sum(int(x["base_generation"]) for x in details) / len(details),
        "prompt_generation_case_acc": sum(int(x["prompt_generation"]) for x in details) / len(details),
        "base_generation_question_acc": sum(int(x) for x in question_base) / len(question_base),
        "prompt_generation_question_acc": sum(int(x) for x in question_prompt) / len(question_prompt),
        "elapsed_seconds": elapsed,
        "mean_seconds_per_case": {k: v / len(details) for k, v in elapsed.items()},
    }
    result = {"config": vars(args), "metrics": metrics, "details": details}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
