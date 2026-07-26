import argparse
import json
from pathlib import Path

from run_real_cqag_experiment_strict import CQAGRealExperiment, ExperimentConfig, matches, parse_int_list, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-ids", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    case_ids = parse_int_list(args.case_ids)
    config = ExperimentConfig(
        model_dir=args.model_dir,
        dataset_path=args.dataset_path,
        seed=args.seed,
        load_in_4bit=True,
        qualified_case_ids=case_ids,
    )
    set_seed(config.seed)
    experiment = CQAGRealExperiment(config)
    wanted = set(case_ids)
    items = [item for item in experiment.dataset if int(item["case_id"]) in wanted]

    details = []
    for item in items:
        facts = " ".join(
            f"{rr['prompt'].format(rr['subject'])} {rr['target_new']['str']}."
            for rr in item["requested_rewrite"]
        )
        preference_successes = []
        generation_successes = []
        outputs = []
        for question in item["questions"]:
            augmented = (
                "You are reasoning inside a hypothetical updated world. Treat the updated facts below as authoritative, "
                "even if they conflict with your prior knowledge. Combine them with unchanged background knowledge to "
                "answer the question. Silently trace every intermediate entity and relation from the question through the "
                "updated facts, then continue through unchanged background facts. Never substitute the old version of an "
                "updated fact. Do not discuss contradictions or the reasoning process.\n"
                f"Updated facts: {facts}\nQuestion: {question}\nAnswer with just the final answer."
            )
            old_score = experiment.answer_logprob(augmented, item["answer"])
            new_score = experiment.answer_logprob(augmented, item["new_answer"])
            pred = experiment.generate_answer(augmented)
            preference_successes.append(new_score > old_score)
            generation_successes.append(matches(pred, item["new_answer"], item.get("new_answer_alias", [])))
            outputs.append({"question": question, "prediction": pred, "old_score": old_score, "new_score": new_score})
        record = {
            "case_id": item["case_id"],
            "preference_success": all(preference_successes),
            "generation_success": all(generation_successes),
            "old_answer": item["answer"],
            "new_answer": item["new_answer"],
            "outputs": outputs,
        }
        details.append(record)
        print(
            f"case={item['case_id']} pref={record['preference_success']} "
            f"gen={record['generation_success']}"
        )

    result = {
        "config": vars(args),
        "metrics": {
            "test_case_count": len(details),
            "graph_prompt_new_preference_acc": sum(int(x["preference_success"]) for x in details) / len(details),
            "graph_prompt_generation_acc": sum(int(x["generation_success"]) for x in details) / len(details),
        },
        "details": details,
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
