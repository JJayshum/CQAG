import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from run_real_cqag_experiment_strict import matches


REFUSAL_RE = re.compile(r"\b(i do not know|i don't know|cannot determine|not enough information|unable to answer)\b", re.I)


def classify(row, item):
    prediction = (row.get("prediction") or "").strip()
    if row.get("selected_source_question") is None and row.get("route_score", 1.0) < 0.35:
        return "route_miss"
    if matches(prediction, item["new_answer"], item.get("new_answer_alias", [])):
        return "correct_new"
    if item["answer"].lower() in prediction.lower() and item["new_answer"].lower() in prediction.lower():
        return "old_and_new_mixed"
    if matches(prediction, item["answer"], item.get("answer_alias", [])):
        return "old_answer_persistence"
    if not prediction:
        return "empty_output"
    if REFUSAL_RE.search(prediction):
        return "refusal_or_uncertainty"
    return "other_wrong_entity_or_reasoning"


def main():
    parser = argparse.ArgumentParser(description="Deterministic failure taxonomy for Value-CQAG outputs.")
    parser.add_argument("--result", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--condition", default="prompt_only")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = json.loads(Path(args.result).read_text(encoding="utf-8"))
    dataset = json.loads(Path(args.dataset_path).read_text(encoding="utf-8"))
    item_by_id = {int(row.get("case_id", i)): row for i, row in enumerate(dataset)}
    rows = result.get("details", {}).get(args.condition, [])
    if not rows:
        raise RuntimeError(f"No rows found for condition {args.condition}")

    classified = []
    for row in rows:
        item = item_by_id[int(row["case_id"])]
        question_index = item["questions"].index(row["question"])
        label = classify(row, item)
        classified.append(
            {
                "case_id": int(row["case_id"]),
                "question": row["question"],
                "question_index": question_index,
                "route_score": row.get("route_score"),
                "route_hit": row.get("selected_source_question") is not None,
                "label": label,
                "prediction": row.get("prediction", ""),
            }
        )

    def normalize_score(value):
        if value is None:
            return "no_route"
        if value < 0.2:
            return "[0,.2)"
        if value < 0.35:
            return "[.2,.35)"
        if value < 0.5:
            return "[.35,.5)"
        return "[.5,1]"

    label_counts = Counter(row["label"] for row in classified)
    by_question_index = defaultdict(Counter)
    by_route_bucket = defaultdict(Counter)
    for row in classified:
        by_question_index[row["question_index"]][row["label"]] += 1
        by_route_bucket[normalize_score(row["route_score"])][row["label"]] += 1

    output = {
        "config": vars(args),
        "total_questions": len(classified),
        "label_counts": dict(label_counts),
        "label_rates": {key: value / len(classified) for key, value in label_counts.items()},
        "by_question_index": {str(key): dict(value) for key, value in by_question_index.items()},
        "by_route_score_bucket": {key: dict(value) for key, value in by_route_bucket.items()},
        "details": classified,
    }
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "details"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
