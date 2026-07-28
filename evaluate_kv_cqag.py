import argparse
import json
from pathlib import Path

import torch

from evaluate_value_router import route
from run_real_cqag_experiment_strict import ExperimentConfig, matches, parse_float_list, parse_int_list, set_seed
from run_value_cqag_question_specific import QuestionSpecificValueCQAG


class KeyValueCQAG(QuestionSpecificValueCQAG):
    def __init__(self, config):
        super().__init__(config)
        self.kv_cache = {}

    def kv_representation(self, prompt, layer_idx):
        cache_key = (prompt, layer_idx)
        if cache_key in self.kv_cache:
            key, value = self.kv_cache[cache_key]
            return key.clone(), value.clone()
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        captured = {}

        def capture(name):
            def hook(_module, _inp, out):
                captured[name] = out[0, -1, :].detach().float().cpu()
            return hook

        attention = self.model.model.layers[layer_idx].self_attn
        handles = [
            attention.k_proj.register_forward_hook(capture("key")),
            attention.v_proj.register_forward_hook(capture("value")),
        ]
        try:
            with torch.no_grad():
                self.model(input_ids=inputs["input_ids"], use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        result = (captured["key"], captured["value"])
        self.kv_cache[cache_key] = result
        return result[0].clone(), result[1].clone()

    def build_question_kv(self, item, layers):
        result = {}
        for question in item["questions"]:
            base = self.build_user_prompt(question)
            augmented = self.build_user_prompt(self.augmented_question(item, question))
            result[question] = {}
            for layer in layers:
                base_k, base_v = self.kv_representation(base, layer)
                aug_k, aug_v = self.kv_representation(augmented, layer)
                result[question][layer] = {"key": aug_k - base_k, "value": aug_v - base_v}
        return result

    def generate_kv(self, question, vectors, alpha, token_window):
        prompt = self.build_user_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        handles = []
        for layer_idx, channels in (vectors or {}).items():
            attention = self.model.model.layers[layer_idx].self_attn
            for channel, module in (("key", attention.k_proj), ("value", attention.v_proj)):
                delta = (alpha * channels[channel]).to(self.model.device)
                state = {"used": False}

                def hook(_module, _inp, out, delta=delta, state=state):
                    if state["used"]:
                        return out
                    state["used"] = True
                    start = max(0, out.shape[1] - token_window)
                    out[:, start:, :] = out[:, start:, :] + delta.to(out.dtype)
                    return out

                handles.append(module.register_forward_hook(hook))
        try:
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens, do_sample=False)
        finally:
            for handle in handles:
                handle.remove()
        return self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def evaluate(experiment, cases, layers, alpha, token_window, threshold):
    details = []
    for case in cases:
        item = case["item"]
        sources = list(case["vectors"])
        for question in item["questions"]:
            remaining = [source for source in sources if source != question]
            selected, score = route(question, remaining, threshold, content_only=True)
            vectors = None if selected is None else {
                layer: case["vectors"][selected][layer] for layer in layers
            }
            prediction = experiment.generate_kv(question, vectors, alpha if vectors else 0.0, token_window)
            success = matches(prediction, item["new_answer"], item.get("new_answer_alias", []))
            details.append({"case_id": int(item["case_id"]), "question": question, "routed": selected is not None,
                            "route_score": score, "success": success, "prediction": prediction})
            print(f"[alpha={alpha}] case={item['case_id']} routed={selected is not None} success={success}", flush=True)
    ids = sorted({row["case_id"] for row in details})
    return {"question_acc": sum(x["success"] for x in details) / len(details),
            "case_acc": sum(all(x["success"] for x in details if x["case_id"] == cid) for cid in ids) / len(ids),
            "route_recall": sum(x["routed"] for x in details) / len(details), "details": details}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--calibration-case-ids", required=True)
    parser.add_argument("--validation-case-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default="20,22,24,26")
    parser.add_argument("--alpha-grid", default="1,2.5,5")
    parser.add_argument("--token-window", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    calibration_ids, validation_ids = parse_int_list(args.calibration_case_ids), parse_int_list(args.validation_case_ids)
    layers = tuple(parse_int_list(args.layers))
    config = ExperimentConfig(model_dir=args.model_dir, dataset_path=args.dataset_path, seed=args.seed,
                              load_in_4bit=True, qualified_case_ids=calibration_ids + validation_ids)
    set_seed(args.seed)
    experiment = KeyValueCQAG(config)
    wanted = set(calibration_ids + validation_ids)
    cases = []
    for item in experiment.dataset:
        if int(item["case_id"]) in wanted:
            cases.append({"item": item, "vectors": experiment.build_question_kv(item, layers)})
            print(f"[vectors] case={item['case_id']}", flush=True)
    by_id = {int(x["item"]["case_id"]): x for x in cases}
    calibration, validation = [by_id[x] for x in calibration_ids], [by_id[x] for x in validation_ids]
    grid = []
    for alpha in parse_float_list(args.alpha_grid):
        result = evaluate(experiment, calibration, layers, alpha, args.token_window, args.threshold)
        grid.append({"alpha": alpha, **{k: v for k, v in result.items() if k != "details"}})
    best = max(grid, key=lambda x: (x["question_acc"], x["case_acc"], -x["alpha"]))
    final = evaluate(experiment, validation, layers, best["alpha"], args.token_window, args.threshold)
    output = {"config": vars(args), "calibration_grid": grid, "best": best, "validation": final}
    temp = Path(args.output + ".tmp")
    temp.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(args.output)
    print(json.dumps({"best": best, "validation": {k: v for k, v in final.items() if k != "details"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
