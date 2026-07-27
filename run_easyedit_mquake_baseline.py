import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

from run_real_cqag_experiment_strict import matches


def answer_logprob(model, tokenizer, question: str, answer: str) -> float:
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False)["input_ids"]
    input_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(model.device)
    prompt_len = prompt_ids.shape[1]
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    count = answer_ids.shape[1]
    return float(gathered[0, prompt_len - 1 : prompt_len - 1 + count].sum().item())


def generate(model, tokenizer, question: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def requests_for(item: dict) -> list[dict]:
    requests = []
    for rewrite in item["requested_rewrite"]:
        requests.append(
            {
                "prompt": rewrite["prompt"],
                "subject": rewrite["subject"],
                "target_new": rewrite["target_new"]["str"],
                "target_true": rewrite["target_true"]["str"],
            }
        )
    return requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--easyedit-dir", required=True)
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--case-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--algorithm", choices=["ROME", "MEMIT"], required=True)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--locality-questions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    sys.path.insert(0, args.easyedit_dir)
    from easyeditor import BaseEditor, MEMITHyperParams, ROMEHyperParams
    from easyeditor.editors.editor import restore_after_edit

    hparams_cls = ROMEHyperParams if args.algorithm == "ROME" else MEMITHyperParams
    hparams = hparams_cls.from_hparams(args.hparams)
    editor = BaseEditor.from_hparams(hparams)
    model, tokenizer = editor.model, editor.tok
    model.eval()

    wanted = {int(x) for x in args.case_ids.split(",") if x.strip()}
    dataset = json.loads(Path(args.dataset_path).read_text(encoding="utf-8"))
    items = [row for i, row in enumerate(dataset) if int(row.get("case_id", i)) in wanted]
    for i, row in enumerate(dataset):
        row.setdefault("case_id", i)
    items = [row for row in dataset if int(row["case_id"]) in wanted]
    rng = random.Random(args.seed)
    details = []

    for item in items:
        cid = int(item["case_id"])
        other_questions = [
            q for row in dataset if int(row["case_id"]) not in wanted for q in row.get("questions", [])
        ]
        locality_questions = rng.sample(other_questions, min(args.locality_questions, len(other_questions)))
        locality_before = [generate(model, tokenizer, q, args.max_new_tokens) for q in locality_questions]
        start = time.perf_counter()
        edited_model, weights_copy = editor.apply_algo(
            model,
            tokenizer,
            requests_for(item),
            hparams,
            copy=False,
            return_orig_weights=True,
            keep_original_weight=True,
        )
        edit_seconds = time.perf_counter() - start
        try:
            preference_hits = []
            predictions = []
            question_hits = []
            for question in item["questions"]:
                old_score = answer_logprob(edited_model, tokenizer, question, item["answer"])
                new_score = answer_logprob(edited_model, tokenizer, question, item["new_answer"])
                preference_hits.append(new_score > old_score)
                pred = generate(edited_model, tokenizer, question, args.max_new_tokens)
                predictions.append(pred)
                question_hits.append(matches(pred, item["new_answer"], item.get("new_answer_alias", [])))
            locality_after = [generate(edited_model, tokenizer, q, args.max_new_tokens) for q in locality_questions]
        finally:
            restore_after_edit(editor, edited_model, weights_copy)

        locality_hits = [before == after for before, after in zip(locality_before, locality_after)]
        row = {
            "case_id": cid,
            "preference_success": all(preference_hits),
            "generation_case_success": all(question_hits),
            "question_hits": question_hits,
            "predictions": predictions,
            "edit_seconds": edit_seconds,
            "locality_hits": locality_hits,
        }
        details.append(row)
        print(
            f"case={cid} pref={row['preference_success']} gen={row['generation_case_success']} "
            f"questions={sum(question_hits)}/{len(question_hits)} locality={sum(locality_hits)}/{len(locality_hits)} "
            f"edit_s={edit_seconds:.2f}",
            flush=True,
        )

    qhits = [hit for row in details for hit in row["question_hits"]]
    lhits = [hit for row in details for hit in row["locality_hits"]]
    metrics = {
        "case_count": len(details),
        "preference_acc": sum(row["preference_success"] for row in details) / len(details),
        "generation_case_acc": sum(row["generation_case_success"] for row in details) / len(details),
        "generation_question_acc": sum(qhits) / len(qhits),
        "exact_output_locality_acc": sum(lhits) / len(lhits),
        "mean_edit_seconds": sum(row["edit_seconds"] for row in details) / len(details),
    }
    result = {"config": vars(args), "metrics": metrics, "details": details}
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
