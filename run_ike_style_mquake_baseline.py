import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

from run_real_cqag_experiment_strict import ExperimentConfig, matches, parse_int_list, set_seed
from run_value_cqag_experiment import ValueCQAGExperiment


def facts(item):
    return " ".join(
        f"{rewrite['prompt'].format(rewrite['subject'])} {rewrite['target_new']['str']}."
        for rewrite in item["requested_rewrite"]
    )


def main():
    parser = argparse.ArgumentParser(description="Transparent IKE-style retrieval baseline for MQuAKE-CF.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--case-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sentence-model", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--train-limit", type=int, default=500)
    parser.add_argument("--locality-questions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    set_seed(args.seed)
    eval_ids = set(parse_int_list(args.case_ids))
    dataset = json.loads(Path(args.dataset_path).read_text(encoding="utf-8"))
    for index, item in enumerate(dataset):
        item.setdefault("case_id", index)
    eval_items = [item for item in dataset if int(item["case_id"]) in eval_ids]
    train_items = [item for item in dataset if int(item["case_id"]) not in eval_ids and int(item["case_id"]) <= 2064]
    rng = random.Random(args.seed)
    rng.shuffle(train_items)
    train_items = train_items[: args.train_limit]

    corpus_texts, corpus_meta = [], []
    for item in train_items:
        for question in item.get("questions", []):
            corpus_texts.append(f"New Fact: {facts(item)}\nPrompt: {question}")
            corpus_meta.append((facts(item), question, item["new_answer"]))
    sentence_model = SentenceTransformer(args.sentence_model, device="cpu")
    corpus_embeddings = sentence_model.encode(corpus_texts, convert_to_tensor=True, normalize_embeddings=True)

    config = ExperimentConfig(
        model_dir=args.model_dir,
        dataset_path=args.dataset_path,
        seed=args.seed,
        load_in_4bit=True,
        qualified_case_ids=sorted(eval_ids),
        max_new_tokens=args.max_new_tokens,
    )
    experiment = ValueCQAGExperiment(config)

    def prompt_for(item, question):
        query = f"New Fact: {facts(item)}\nPrompt: {question}"
        query_embedding = sentence_model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        scores = util.dot_score(query_embedding, corpus_embeddings)[0]
        top_indices = torch.topk(scores, k=min(args.k, len(corpus_meta))).indices.tolist()
        demonstrations = []
        for index in top_indices:
            fact, demo_question, answer = corpus_meta[index]
            demonstrations.append(
                f"New Fact: {fact}\nQuestion: {demo_question}\nAnswer: {answer}\n"
            )
        return (
            "Use the following edited-fact demonstrations as in-context examples. "
            "Answer with only the final answer.\n\n"
            + "\n".join(demonstrations)
            + f"\nCurrent updated facts: {facts(item)}\nQuestion: {question}\nAnswer:"
        )

    details = []
    start = time.perf_counter()
    for item in eval_items:
        question_hits, preference_hits, predictions = [], [], []
        for question in item["questions"]:
            prompt = prompt_for(item, question)
            old_score = experiment.answer_logprob(prompt, item["answer"])
            new_score = experiment.answer_logprob(prompt, item["new_answer"])
            prediction = experiment.generate_answer(prompt)
            preference_hits.append(new_score > old_score)
            question_hits.append(matches(prediction, item["new_answer"], item.get("new_answer_alias", [])))
            predictions.append(prediction)
        details.append(
            {
                "case_id": int(item["case_id"]),
                "preference_success": all(preference_hits),
                "generation_case_success": all(question_hits),
                "question_hits": question_hits,
                "predictions": predictions,
            }
        )
        print(f"[eval] case={item['case_id']} gen={sum(question_hits)}/{len(question_hits)}", flush=True)

    unrelated = [
        (int(item["case_id"]), question, item["answer"])
        for item in dataset
        if int(item["case_id"]) not in eval_ids
        for question in item.get("questions", [])
    ]
    unrelated = rng.sample(unrelated, min(args.locality_questions, len(unrelated)))
    locality_rows = []
    for item in eval_items:
        for case_id, question, answer in unrelated:
            base = experiment.generate_answer(question)
            edited = experiment.generate_answer(prompt_for(item, question))
            locality_rows.append(
                {
                    "edited_case_id": int(item["case_id"]),
                    "unrelated_case_id": case_id,
                    "question": question,
                    "base": base,
                    "edited": edited,
                    "exact_output_preserved": base == edited,
                    "base_correct": matches(base, answer, []),
                    "edited_correct": matches(edited, answer, []),
                }
            )

    questions = [hit for row in details for hit in row["question_hits"]]
    metrics = {
        "case_count": len(details),
        "preference_acc": sum(row["preference_success"] for row in details) / len(details),
        "generation_case_acc": sum(row["generation_case_success"] for row in details) / len(details),
        "generation_question_acc": sum(questions) / len(questions),
        "exact_output_locality_acc": sum(row["exact_output_preserved"] for row in locality_rows) / len(locality_rows),
        "semantic_locality_acc": sum(row["base_correct"] == row["edited_correct"] for row in locality_rows) / len(locality_rows),
        "mean_seconds_per_case": (time.perf_counter() - start) / len(details),
        "train_case_count": len(train_items),
        "retrieval_k": args.k,
    }
    output = {"config": vars(args), "metrics": metrics, "details": details, "locality": locality_rows}
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
