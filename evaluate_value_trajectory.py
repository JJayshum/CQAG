import argparse
import json
import random
from pathlib import Path

import torch

from evaluate_value_router import route
from run_real_cqag_experiment_strict import ExperimentConfig, matches, parse_float_list, parse_int_list, set_seed
from run_value_cqag_question_specific import QuestionSpecificValueCQAG


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


class TrajectoryValueCQAG(QuestionSpecificValueCQAG):
    def __init__(self, config):
        super().__init__(config)
        self.trajectory_cache = {}

    def value_trajectory(self, prompt, layer_idx, token_window):
        key = (prompt, layer_idx, token_window)
        if key in self.trajectory_cache:
            return self.trajectory_cache[key].clone()
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        captured = []

        def capture(_module, _inp, out):
            captured.append(out[0, -token_window:, :].detach().float().cpu())

        handle = self.model.model.layers[layer_idx].self_attn.v_proj.register_forward_hook(capture)
        try:
            with torch.no_grad():
                self.model(input_ids=inputs["input_ids"], use_cache=False)
        finally:
            handle.remove()
        trajectory = captured[0]
        self.trajectory_cache[key] = trajectory
        return trajectory.clone()

    def build_question_trajectories(self, item, layers, token_window):
        result = {}
        for question in item["questions"]:
            base_prompt = self.build_user_prompt(question)
            augmented_prompt = self.build_user_prompt(self.augmented_question(item, question))
            result[question] = {
                layer: self.value_trajectory(augmented_prompt, layer, token_window)
                - self.value_trajectory(base_prompt, layer, token_window)
                for layer in layers
            }
        return result

    def generate_trajectory(self, question, trajectories, alpha):
        prompt = self.build_user_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        handles = []
        if trajectories and alpha:
            for layer_idx, trajectory in trajectories.items():
                delta = (alpha * trajectory).to(self.model.device)
                state = {"used": False}

                def hook(_module, _inp, out, delta=delta, state=state):
                    if state["used"]:
                        return out
                    state["used"] = True
                    count = min(out.shape[1], delta.shape[0])
                    out[:, -count:, :] = out[:, -count:, :] + delta[-count:].to(out.dtype)
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
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()


def evaluate(experiment, cases, layers, alpha, threshold):
    details = []
    for case in cases:
        item = case["item"]
        source_questions = list(case["trajectories"])
        for question in item["questions"]:
            remaining = [source for source in source_questions if source != question]
            selected, score = route(question, remaining, threshold, content_only=True)
            trajectories = None
            if selected is not None:
                trajectories = {
                    layer: case["trajectories"][selected][layer]
                    for layer in layers
                }
            prediction = experiment.generate_trajectory(
                question,
                trajectories,
                alpha if trajectories is not None else 0.0,
            )
            success = matches(prediction, item["new_answer"], item.get("new_answer_alias", []))
            details.append(
                {
                    "case_id": int(item["case_id"]),
                    "question": question,
                    "selected_source_question": selected,
                    "route_score": score,
                    "success": success,
                    "prediction": prediction,
                }
            )
            print(
                f"[eval alpha={alpha}] case={item['case_id']} routed={selected is not None} success={success}",
                flush=True,
            )
    case_ids = sorted({row["case_id"] for row in details})
    return {
        "question_acc": mean(row["success"] for row in details),
        "case_acc": mean(
            all(row["success"] for row in details if row["case_id"] == case_id)
            for case_id in case_ids
        ),
        "route_recall": mean(row["selected_source_question"] is not None for row in details),
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Token-aligned Value trajectory development experiment.")
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

    calibration_ids = parse_int_list(args.calibration_case_ids)
    validation_ids = parse_int_list(args.validation_case_ids)
    all_ids = calibration_ids + validation_ids
    layers = tuple(parse_int_list(args.layers))
    config = ExperimentConfig(
        model_dir=args.model_dir,
        dataset_path=args.dataset_path,
        seed=args.seed,
        load_in_4bit=True,
        qualified_case_ids=all_ids,
    )
    set_seed(args.seed)
    experiment = TrajectoryValueCQAG(config)
    wanted = set(all_ids)
    cases = []
    for item in experiment.dataset:
        if int(item["case_id"]) not in wanted:
            continue
        cases.append(
            {
                "item": item,
                "trajectories": experiment.build_question_trajectories(
                    item, layers, args.token_window
                ),
            }
        )
        print(f"[trajectories] case={item['case_id']}", flush=True)
    by_id = {int(case["item"]["case_id"]): case for case in cases}
    calibration = [by_id[case_id] for case_id in calibration_ids]
    validation = [by_id[case_id] for case_id in validation_ids]

    grid = []
    calibration_details = {}
    for alpha in parse_float_list(args.alpha_grid):
        result = evaluate(experiment, calibration, layers, alpha, args.threshold)
        grid.append({key: value for key, value in result.items() if key != "details"} | {"alpha": alpha})
        calibration_details[str(alpha)] = result["details"]
    best = max(grid, key=lambda row: (row["question_acc"], row["case_acc"], -row["alpha"]))
    validation_result = evaluate(
        experiment, validation, layers, float(best["alpha"]), args.threshold
    )
    output = {
        "config": vars(args),
        "calibration_grid": grid,
        "best": best,
        "validation": validation_result,
        "calibration_details": calibration_details,
    }
    temporary = Path(args.output + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "best": best,
                "validation": {key: value for key, value in validation_result.items() if key != "details"},
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
