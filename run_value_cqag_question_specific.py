import argparse
import json
import random
from pathlib import Path
from typing import Dict

import torch

from run_real_cqag_experiment_strict import ExperimentConfig, matches, parse_float_list, parse_int_list, set_seed
from run_value_cqag_experiment import ValueCQAGExperiment, parse_layer_sets


class QuestionSpecificValueCQAG(ValueCQAGExperiment):
    def build_question_vectors(self, item: dict, layers: list[int]) -> Dict[str, Dict[int, torch.Tensor]]:
        result = {}
        for question in item["questions"]:
            base_prompt = self.build_user_prompt(question)
            augmented_prompt = self.build_user_prompt(self.augmented_question(item, question))
            result[question] = {
                layer: self.value_representation(augmented_prompt, layer)
                - self.value_representation(base_prompt, layer)
                for layer in layers
            }
        return result

    def case_success_question_specific(
        self,
        item: dict,
        vectors: Dict[str, Dict[int, torch.Tensor]],
        alpha: float,
        token_window: int,
    ) -> bool:
        for question in item["questions"]:
            old_score = self.answer_logprob_value(
                question, item["answer"], vectors[question], alpha, token_window
            )
            new_score = self.answer_logprob_value(
                question, item["new_answer"], vectors[question], alpha, token_window
            )
            if new_score <= old_score:
                return False
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Question-specific no-leak Value-CQAG development experiment.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--case-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calib-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--layer-sets", default="20,22,24;22,24,26;20,22,24,26")
    parser.add_argument("--alpha-grid", default="2.5,5,10,20")
    parser.add_argument("--token-windows", default="4,8,16")
    parser.add_argument(
        "--eval-all",
        action="store_true",
        help="Evaluate every supplied case with one frozen configuration; disables calibration.",
    )
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
    experiment = QuestionSpecificValueCQAG(config)
    wanted = set(case_ids)
    items = [item for item in experiment.dataset if int(item["case_id"]) in wanted]
    random.Random(args.seed).shuffle(items)
    calib_items = [] if args.eval_all else items[: args.calib_size]
    validation_items = items if args.eval_all else items[args.calib_size :]
    layer_sets = parse_layer_sets(args.layer_sets)
    alpha_grid = parse_float_list(args.alpha_grid)
    token_windows = parse_int_list(args.token_windows)
    all_layers = sorted({layer for group in layer_sets for layer in group})
    vectors = {
        int(item["case_id"]): experiment.build_question_vectors(item, all_layers)
        for item in items
    }

    grid = []
    best = {"score": -1.0}
    if args.eval_all:
        if len(layer_sets) != 1 or len(alpha_grid) != 1 or len(token_windows) != 1:
            raise ValueError("--eval-all requires exactly one layer set, alpha, and token window")
        best = {
            "layers": list(layer_sets[0]),
            "alpha": alpha_grid[0],
            "token_window": token_windows[0],
            "score": None,
        }
    else:
        for layers in layer_sets:
            for alpha in alpha_grid:
                for window in token_windows:
                    hits = []
                    for item in calib_items:
                        question_vectors = {
                            q: {layer: vectors[int(item["case_id"])][q][layer] for layer in layers}
                            for q in item["questions"]
                        }
                        hits.append(
                            experiment.case_success_question_specific(item, question_vectors, alpha, window)
                        )
                    score = sum(hits) / len(hits)
                    row = {"layers": list(layers), "alpha": alpha, "token_window": window, "score": score}
                    grid.append(row)
                    print(f"[calib] {row}", flush=True)
                    if score > best["score"]:
                        best = row

    layers = tuple(best["layers"])
    alpha = float(best["alpha"])
    window = int(best["token_window"])
    details = []
    for item in validation_items:
        cid = int(item["case_id"])
        q_vectors = {
            q: {layer: vectors[cid][q][layer] for layer in layers}
            for q in item["questions"]
        }
        preference = experiment.case_success_question_specific(item, q_vectors, alpha, window)
        predictions = [
            experiment.generate_value(q, q_vectors[q], alpha, window) for q in item["questions"]
        ]
        question_hits = [
            matches(pred, item["new_answer"], item.get("new_answer_alias", [])) for pred in predictions
        ]
        details.append(
            {
                "case_id": cid,
                "preference_success": preference,
                "generation_case_success": all(question_hits),
                "question_hits": question_hits,
                "predictions": predictions,
            }
        )
        print(
            f"[validation] case={cid} pref={preference} gen={all(question_hits)} "
            f"question_hits={sum(question_hits)}/{len(question_hits)}",
            flush=True,
        )

    question_hits = [hit for row in details for hit in row["question_hits"]]
    metrics = {
        "calib_case_count": len(calib_items),
        "validation_case_count": len(validation_items),
        "preference_acc": sum(int(x["preference_success"]) for x in details) / len(details),
        "generation_case_acc": sum(int(x["generation_case_success"]) for x in details) / len(details),
        "generation_question_acc": sum(int(x) for x in question_hits) / len(question_hits),
    }
    result = {
        "config": vars(args),
        "calib_case_ids": [int(item["case_id"]) for item in calib_items],
        "validation_case_ids": [int(item["case_id"]) for item in validation_items],
        "grid": grid,
        "best": best,
        "metrics": metrics,
        "details": details,
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best, "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
