import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

from run_real_cqag_experiment_strict import (
    CQAGRealExperiment,
    ExperimentConfig,
    matches,
    parse_float_list,
    parse_int_list,
    set_seed,
)


DEFAULT_CASE_IDS = (
    1, 13, 14, 18, 20, 21, 25, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 39, 49, 52,
    54, 63, 64, 66, 82, 84, 87, 93, 100, 101, 109, 110, 116, 118, 123, 125, 127, 129,
    134, 135,
)


class ValueCQAGExperiment(CQAGRealExperiment):
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.value_cache: Dict[Tuple[str, int], torch.Tensor] = {}

    @staticmethod
    def updated_facts(item: dict) -> str:
        return " ".join(
            f"{rr['prompt'].format(rr['subject'])} {rr['target_new']['str']}."
            for rr in item["requested_rewrite"]
        )

    def augmented_question(self, item: dict, question: str) -> str:
        return (
            "You are reasoning inside a hypothetical updated world. Treat the updated facts below as authoritative, "
            "even if they conflict with prior knowledge. Combine them with unchanged background knowledge. "
            "Do not discuss contradictions.\n"
            f"Updated facts: {self.updated_facts(item)}\nQuestion: {question}\n"
            "Answer with just the final answer."
        )

    def value_representation(self, prompt: str, layer_idx: int) -> torch.Tensor:
        key = (prompt, layer_idx)
        if key in self.value_cache:
            return self.value_cache[key].clone()
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        captured: List[torch.Tensor] = []

        def capture(_module, _inp, out):
            captured.append(out[0, -1, :].detach().float().cpu())

        handle = self.model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(capture)
        try:
            with torch.no_grad():
                self.model(input_ids=inputs["input_ids"], use_cache=False)
        finally:
            handle.remove()
        rep = captured[0]
        self.value_cache[key] = rep
        return rep.clone()

    def build_value_vectors(self, item: dict, layers: Iterable[int]) -> Dict[int, torch.Tensor]:
        vectors: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
        for question in item["questions"]:
            base_prompt = self.build_user_prompt(question)
            augmented_prompt = self.build_user_prompt(self.augmented_question(item, question))
            for layer in layers:
                delta = self.value_representation(augmented_prompt, layer) - self.value_representation(base_prompt, layer)
                vectors[layer].append(delta)
        return {layer: torch.stack(parts).mean(dim=0) for layer, parts in vectors.items()}

    def answer_logprob_value(
        self,
        question: str,
        answer: str,
        vectors: Dict[int, torch.Tensor] | None = None,
        alpha: float = 0.0,
        token_window: int = 1,
    ) -> float:
        prompt = self.build_user_prompt(question)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer, return_tensors="pt", add_special_tokens=False)["input_ids"]
        input_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(self.model.device)
        prompt_len = prompt_ids.shape[1]
        handles = []
        if vectors and alpha:
            for layer_idx, vector in vectors.items():
                vec = (alpha * vector).to(self.model.device)

                def hook(_module, _inp, out, vec=vec):
                    start = max(0, prompt_len - token_window)
                    out[:, start:prompt_len, :] = out[:, start:prompt_len, :] + vec.to(out.dtype)
                    return out

                handles.append(self.model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(hook))
        try:
            with torch.no_grad():
                logits = self.model(input_ids=input_ids, use_cache=False).logits[:, :-1, :]
        finally:
            for handle in handles:
                handle.remove()
        targets = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        count = answer_ids.shape[1]
        return float(gathered[0, prompt_len - 1 : prompt_len - 1 + count].sum().item())

    def case_success_value(
        self,
        item: dict,
        vectors: Dict[int, torch.Tensor] | None,
        alpha: float,
        token_window: int = 1,
    ) -> bool:
        for question in item["questions"]:
            old_score = self.answer_logprob_value(question, item["answer"], vectors, alpha, token_window)
            new_score = self.answer_logprob_value(question, item["new_answer"], vectors, alpha, token_window)
            if new_score <= old_score:
                return False
        return True

    def generate_value(
        self,
        question: str,
        vectors: Dict[int, torch.Tensor] | None,
        alpha: float,
        token_window: int = 1,
        generation_steps: int = 0,
    ) -> str:
        prompt = self.build_user_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        handles = []
        hook_calls = {layer: 0 for layer in (vectors or {})}
        if vectors and alpha:
            for layer_idx, vector in vectors.items():
                vec = (alpha * vector).to(self.model.device)

                def hook(_module, _inp, out, layer_idx=layer_idx, vec=vec):
                    call_index = hook_calls[layer_idx]
                    if call_index == 0:
                        start = max(0, out.shape[1] - token_window)
                        out[:, start:, :] = out[:, start:, :] + vec.to(out.dtype)
                    elif call_index <= generation_steps:
                        out[:, :, :] = out[:, :, :] + vec.to(out.dtype)
                    hook_calls[layer_idx] += 1
                    return out

                handles.append(self.model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(hook))
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                )
        finally:
            for handle in handles:
                handle.remove()
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def parse_layer_sets(text: str) -> List[Tuple[int, ...]]:
    return [parse_int_list(group) for group in text.split(";") if group.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="No-leak multi-layer Value-channel CQAG experiment.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-ids", default=",".join(map(str, DEFAULT_CASE_IDS)))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dev-size", type=int, default=10)
    parser.add_argument("--layer-sets", default="20;22;24;26;20,22;22,24;24,26;20,22,24;22,24,26;20,22,24,26")
    parser.add_argument("--alpha-grid", default="0.25,0.5,1.0,2.0,4.0")
    parser.add_argument("--token-windows", default="1,4,8,16")
    args = parser.parse_args()

    case_ids = parse_int_list(args.case_ids)
    config = ExperimentConfig(
        model_dir=args.model_dir,
        dataset_path=args.dataset_path,
        seed=args.seed,
        load_in_4bit=True,
        qualified_case_ids=case_ids,
    )
    set_seed(args.seed)
    experiment = ValueCQAGExperiment(config)
    wanted = set(case_ids)
    items = [item for item in experiment.dataset if int(item["case_id"]) in wanted]
    random.Random(args.seed).shuffle(items)
    dev_items = items[: args.dev_size]
    test_items = items[args.dev_size :]
    layer_sets = parse_layer_sets(args.layer_sets)
    alpha_grid = parse_float_list(args.alpha_grid)
    token_windows = parse_int_list(args.token_windows)
    all_layers = sorted({layer for layer_set in layer_sets for layer in layer_set})
    vectors = {int(item["case_id"]): experiment.build_value_vectors(item, all_layers) for item in items}

    calibration = []
    best = {"score": -1.0, "layers": None, "alpha": None}
    for layer_set in layer_sets:
        for alpha in alpha_grid:
            for token_window in token_windows:
                successes = []
                for item in dev_items:
                    item_vectors = {layer: vectors[int(item["case_id"])][layer] for layer in layer_set}
                    successes.append(experiment.case_success_value(item, item_vectors, alpha, token_window))
                score = sum(successes) / len(successes)
                row = {"layers": list(layer_set), "alpha": alpha, "token_window": token_window, "score": score}
                calibration.append(row)
                print(
                    f"[calib] layers={layer_set} alpha={alpha} window={token_window} score={score:.3f}",
                    flush=True,
                )
                if score > best["score"]:
                    best = row

    chosen_layers = tuple(best["layers"])
    chosen_alpha = float(best["alpha"])
    chosen_window = int(best["token_window"])
    details = []
    for index, item in enumerate(test_items):
        cid = int(item["case_id"])
        item_vectors = {layer: vectors[cid][layer] for layer in chosen_layers}
        random_vectors = {}
        for layer, vector in item_vectors.items():
            random_vector = torch.randn_like(vector)
            random_vector = random_vector / random_vector.norm().clamp_min(1e-8) * vector.norm()
            random_vectors[layer] = random_vector
        mismatch = test_items[(index + 1) % len(test_items)]
        mismatch_vectors = {layer: vectors[int(mismatch["case_id"])][layer] for layer in chosen_layers}
        value_success = experiment.case_success_value(item, item_vectors, chosen_alpha, chosen_window)
        random_success = experiment.case_success_value(item, random_vectors, chosen_alpha, chosen_window)
        mismatch_preserves_old = not experiment.case_success_value(
            item, mismatch_vectors, chosen_alpha, chosen_window
        )
        predictions = [
            experiment.generate_value(q, item_vectors, chosen_alpha, chosen_window) for q in item["questions"]
        ]
        generation_success = all(matches(pred, item["new_answer"], item.get("new_answer_alias", [])) for pred in predictions)
        question_hits = [matches(pred, item["new_answer"], item.get("new_answer_alias", [])) for pred in predictions]
        details.append(
            {
                "case_id": cid,
                "value_success": value_success,
                "random_success": random_success,
                "mismatch_preserves_old": mismatch_preserves_old,
                "generation_success": generation_success,
                "question_hits": question_hits,
                "predictions": predictions,
                "old_answer": item["answer"],
                "new_answer": item["new_answer"],
            }
        )
        print(
            f"[test] case={cid} value={value_success} random={random_success} "
            f"locality={mismatch_preserves_old} gen={generation_success}",
            flush=True,
        )

    question_hits = [hit for row in details for hit in row["question_hits"]]
    metrics = {
        "dev_case_count": len(dev_items),
        "test_case_count": len(test_items),
        "value_new_preference_acc": sum(int(x["value_success"]) for x in details) / len(details),
        "random_new_preference_acc": sum(int(x["random_success"]) for x in details) / len(details),
        "mismatched_locality_acc": sum(int(x["mismatch_preserves_old"]) for x in details) / len(details),
        "value_generation_case_acc": sum(int(x["generation_success"]) for x in details) / len(details),
        "value_generation_question_acc": sum(int(x) for x in question_hits) / len(question_hits),
    }
    result = {
        "config": vars(args),
        "dev_case_ids": [int(item["case_id"]) for item in dev_items],
        "test_case_ids": [int(item["case_id"]) for item in test_items],
        "calibration": calibration,
        "best": best,
        "metrics": metrics,
        "details": details,
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
