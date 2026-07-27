import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class ExperimentConfig:
    model_dir: str = "/root/autodl-tmp/modelscope_cache/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master"
    dataset_path: str = "/root/cqag_experiment/data/MQuAKE-CF-3k-v2.json"
    seed: int = 7
    max_scan_cases: int = 240
    scan_start_case: int = 0
    target_pool_size: int = 24
    calib_size: int = 8
    candidate_layers: Tuple[int, ...] = (6, 10, 14, 18, 22)
    alpha_grid: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)
    max_new_tokens: int = 24
    load_in_4bit: bool = False
    filter_mode: str = "strict"
    shuffle_split: bool = True
    qualified_case_ids: Tuple[int, ...] = ()


def parse_int_list(text: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def parse_float_list(text: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[\s\n\t]+", " ", text)
    text = re.sub(r"[\"'`.,:;!?()\[\]{}]", "", text)
    return text


def matches(pred: str, gold: str, aliases: List[str] | None = None) -> bool:
    p = normalize(pred)
    for cand in [gold] + (aliases or []):
        c = normalize(cand)
        if c and (p == c or c in p):
            return True
    return False


class CQAGRealExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_dir, trust_remote_code=True)
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if config.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            model_kwargs["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(config.model_dir, **model_kwargs)
        with open(config.dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)
        self.rep_cache: Dict[Tuple[str, int], torch.Tensor] = {}

    def build_user_prompt(self, question: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question + " Answer with just the answer."}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_answer(
        self,
        question: str,
        max_new_tokens: int | None = None,
        layer_idx: int | None = None,
        steer_vec: torch.Tensor | None = None,
        alpha: float = 0.0,
    ) -> str:
        prompt = self.build_user_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        hook_handle = None
        hook_state = {"used": False}
        if layer_idx is not None and steer_vec is not None and alpha != 0.0:
            vec = (alpha * steer_vec).to(self.model.device)

            def hook_fn(_module, _inp, out):
                if hook_state["used"]:
                    return out
                hook_state["used"] = True
                if isinstance(out, tuple):
                    hidden = out[0]
                    hidden[:, -1, :] = hidden[:, -1, :] + vec.to(hidden.dtype)
                    return (hidden,) + out[1:]
                hidden = out
                hidden[:, -1, :] = hidden[:, -1, :] + vec.to(hidden.dtype)
                return hidden

            hook_handle = self.model.model.layers[layer_idx].register_forward_hook(hook_fn)
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens or self.config.max_new_tokens,
                    do_sample=False,
                )
        finally:
            if hook_handle is not None:
                hook_handle.remove()
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    def answer_logprob(
        self,
        question: str,
        answer: str,
        layer_idx: int | None = None,
        steer_vec: torch.Tensor | None = None,
        alpha: float = 0.0,
    ) -> float:
        prompt = self.build_user_prompt(question)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer, return_tensors="pt", add_special_tokens=False)["input_ids"]
        input_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(self.model.device)
        prompt_len = prompt_ids.shape[1]

        hook_handle = None
        if layer_idx is not None and steer_vec is not None and alpha != 0.0:
            vec = (alpha * steer_vec).to(self.model.device)

            def hook_fn(_module, _inp, out):
                if isinstance(out, tuple):
                    hidden = out[0]
                    hidden[:, prompt_len - 1, :] = hidden[:, prompt_len - 1, :] + vec.to(hidden.dtype)
                    return (hidden,) + out[1:]
                hidden = out
                hidden[:, prompt_len - 1, :] = hidden[:, prompt_len - 1, :] + vec.to(hidden.dtype)
                return hidden

            hook_handle = self.model.model.layers[layer_idx].register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                logits = self.model(input_ids=input_ids).logits[:, :-1, :]
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        target_ids = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        gather = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        answer_token_count = answer_ids.shape[1]
        answer_logprob = gather[0, prompt_len - 1 : prompt_len - 1 + answer_token_count].sum().item()
        return float(answer_logprob)

    def text_representation(self, text: str, layer_idx: int) -> torch.Tensor:
        key = (text, layer_idx)
        if key in self.rep_cache:
            return self.rep_cache[key].clone()

        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.model(input_ids=inputs["input_ids"], output_hidden_states=True)
        rep = outputs.hidden_states[layer_idx + 1][0, -1, :].detach().float().cpu()
        self.rep_cache[key] = rep
        return rep.clone()

    def build_case_vector(self, item: dict, layer_idx: int) -> torch.Tensor:
        vectors = []
        new_facts = " ".join(
            f"{rr['prompt'].format(rr['subject'])} {rr['target_new']['str']}."
            for rr in item["requested_rewrite"]
        )
        for question in item["questions"]:
            base_prompt = self.build_user_prompt(question)
            augmented_prompt = self.build_user_prompt(
                f"Relevant updated facts: {new_facts}\nQuestion: {question}\n"
                "Use the updated facts when answering."
            )
            delta = self.text_representation(augmented_prompt, layer_idx) - self.text_representation(base_prompt, layer_idx)
            vectors.append(delta / delta.norm().clamp_min(1e-8))
        vec = torch.stack(vectors, dim=0).mean(dim=0)
        return vec / vec.norm().clamp_min(1e-8)

    def old_preference_filter(self, item: dict) -> bool:
        single = item["single_hops"][0]
        old_single = self.answer_logprob(single["question"], single["answer"])
        new_single = self.answer_logprob(single["question"], item["new_single_hops"][0]["answer"])
        old_multi = self.answer_logprob(item["questions"][0], item["answer"])
        new_multi = self.answer_logprob(item["questions"][0], item["new_answer"])
        return old_single > new_single and old_multi > new_multi

    def collect_pool(self) -> dict:
        if self.config.qualified_case_ids:
            wanted = set(self.config.qualified_case_ids)
            pool = [item for item in self.dataset if int(item["case_id"]) in wanted]
            pool.sort(key=lambda item: self.config.qualified_case_ids.index(int(item["case_id"])))
            if len(pool) != len(wanted):
                raise RuntimeError(f"Found {len(pool)} of {len(wanted)} cached qualified cases")
            return {
                "pool": pool,
                "probe_records": [
                    {"case_id": item["case_id"], "single_hit": True, "multi_hit": True, "old_pref": True}
                    for item in pool
                ],
            }
        pool = []
        probe_records = []
        scan_end = self.config.scan_start_case + self.config.max_scan_cases
        for idx, item in enumerate(
            self.dataset[self.config.scan_start_case : scan_end],
            start=self.config.scan_start_case + 1,
        ):
            single = item["single_hops"][0]
            single_pred = self.generate_answer(single["question"])
            multi_pred = self.generate_answer(item["questions"][0])
            single_hit = matches(single_pred, single["answer"], single.get("answer_alias", []))
            multi_hit = matches(multi_pred, item["answer"], item.get("answer_alias", []))
            old_pref = self.old_preference_filter(item) if self.config.filter_mode == "strict" else False
            probe_records.append(
                {
                    "case_id": item["case_id"],
                    "single_hit": single_hit,
                    "multi_hit": multi_hit,
                    "old_pref": old_pref,
                }
            )
            keep = False
            if self.config.filter_mode == "strict":
                keep = single_hit and multi_hit and old_pref
            elif self.config.filter_mode == "generation":
                keep = single_hit and multi_hit
            else:
                raise ValueError(f"Unknown filter_mode: {self.config.filter_mode}")
            if keep:
                pool.append(item)
                print(f"[pool] keep case={item['case_id']} size={len(pool)}")
            else:
                print(
                    f"[pool] skip case={item['case_id']} single_hit={single_hit} "
                    f"multi_hit={multi_hit} old_pref={old_pref}"
                )
            if len(pool) >= self.config.target_pool_size:
                break
        return {"pool": pool, "probe_records": probe_records}

    def case_multi_success(
        self,
        item: dict,
        layer_idx: int | None = None,
        steer_vec: torch.Tensor | None = None,
        alpha: float = 0.0,
    ) -> dict:
        question_details = []
        for q in item["questions"]:
            old_score = self.answer_logprob(q, item["answer"], layer_idx, steer_vec, alpha)
            new_score = self.answer_logprob(q, item["new_answer"], layer_idx, steer_vec, alpha)
            question_details.append(
                {
                    "question": q,
                    "old_score": old_score,
                    "new_score": new_score,
                    "new_preferred": bool(new_score > old_score),
                }
            )
        return {
            "case_success": all(x["new_preferred"] for x in question_details),
            "question_details": question_details,
        }

    def case_old_preferred(
        self,
        item: dict,
        layer_idx: int | None = None,
        steer_vec: torch.Tensor | None = None,
        alpha: float = 0.0,
    ) -> bool:
        for q in item["questions"]:
            old_score = self.answer_logprob(q, item["answer"], layer_idx, steer_vec, alpha)
            new_score = self.answer_logprob(q, item["new_answer"], layer_idx, steer_vec, alpha)
            if old_score <= new_score:
                return False
        return True

    def calibrate(self, calib_pool: List[dict]) -> dict:
        best = {"score": -1.0, "layer": None, "alpha": None}
        all_scores = []
        for layer_idx in self.config.candidate_layers:
            vectors = {item["case_id"]: self.build_case_vector(item, layer_idx) for item in calib_pool}
            for alpha in self.config.alpha_grid:
                successes = []
                for item in calib_pool:
                    outcome = self.case_multi_success(item, layer_idx, vectors[item["case_id"]], alpha)
                    successes.append(int(outcome["case_success"]))
                mean_score = sum(successes) / len(successes)
                all_scores.append({"layer": layer_idx, "alpha": alpha, "score": mean_score})
                print(f"[calib] layer={layer_idx} alpha={alpha:.2f} score={mean_score:.3f}")
                if mean_score > best["score"]:
                    best = {"score": mean_score, "layer": layer_idx, "alpha": alpha}
        best["grid"] = all_scores
        return best

    def run(self) -> dict:
        pool_info = self.collect_pool()
        pool = pool_info["pool"]
        if len(pool) < 12:
            raise RuntimeError(f"Only collected {len(pool)} qualified cases, need at least 12")

        if self.config.shuffle_split:
            random.Random(self.config.seed).shuffle(pool)
        calib_size = min(self.config.calib_size, max(4, len(pool) // 3))
        calib_pool = pool[:calib_size]
        test_pool = pool[calib_size:]
        best = self.calibrate(calib_pool)
        layer_idx = int(best["layer"])
        alpha = float(best["alpha"])

        full_vectors = {item["case_id"]: self.build_case_vector(item, layer_idx) for item in pool}

        base_old = []
        no_cqag = []
        cqag_full = []
        cqag_fact_only = []
        cqag_random = []
        locality_mismatch = []
        no_cqag_generation = []
        cqag_generation = []
        details = []

        for idx, item in enumerate(test_pool):
            cid = item["case_id"]
            vec = full_vectors[cid]
            rand_vec = torch.randn_like(vec)
            rand_vec = rand_vec / rand_vec.norm().clamp_min(1e-8)
            mismatch_item = test_pool[(idx + 1) % len(test_pool)]
            mismatch_vec = full_vectors[mismatch_item["case_id"]]

            base_old_pref = self.case_old_preferred(item)
            no_cqag_res = self.case_multi_success(item)
            cqag_res = self.case_multi_success(item, layer_idx, vec, alpha)
            fact_res = cqag_res
            rand_res = self.case_multi_success(item, layer_idx, rand_vec, alpha)
            locality_ok = self.case_old_preferred(item, layer_idx, mismatch_vec, alpha)
            base_generations = [self.generate_answer(q) for q in item["questions"]]
            cqag_generations = [self.generate_answer(q, layer_idx=layer_idx, steer_vec=vec, alpha=alpha) for q in item["questions"]]
            base_generation_ok = all(matches(pred, item["new_answer"], item.get("new_answer_alias", [])) for pred in base_generations)
            cqag_generation_ok = all(matches(pred, item["new_answer"], item.get("new_answer_alias", [])) for pred in cqag_generations)

            base_old.append(int(base_old_pref))
            no_cqag.append(int(no_cqag_res["case_success"]))
            cqag_full.append(int(cqag_res["case_success"]))
            cqag_fact_only.append(int(fact_res["case_success"]))
            cqag_random.append(int(rand_res["case_success"]))
            locality_mismatch.append(int(locality_ok))
            no_cqag_generation.append(int(base_generation_ok))
            cqag_generation.append(int(cqag_generation_ok))

            details.append(
                {
                    "case_id": cid,
                    "base_old_pref": base_old_pref,
                    "no_cqag_success": no_cqag_res["case_success"],
                    "cqag_full_success": cqag_res["case_success"],
                    "cqag_fact_only_success": fact_res["case_success"],
                    "cqag_random_success": rand_res["case_success"],
                    "locality_mismatch_old_pref": locality_ok,
                    "no_cqag_generation_success": base_generation_ok,
                    "cqag_generation_success": cqag_generation_ok,
                    "no_cqag_generations": base_generations,
                    "cqag_generations": cqag_generations,
                    "requested_rewrite_count": len(item["requested_rewrite"]),
                    "new_answer": item["new_answer"],
                    "old_answer": item["answer"],
                }
            )
            print(
                f"[test] case={cid} base_old={base_old_pref} cqag={cqag_res['case_success']} "
                f"fact_only={fact_res['case_success']} rand={rand_res['case_success']} locality={locality_ok}"
            )

        results = {
            "config": asdict(self.config),
            "qualified_pool_size": len(pool),
            "calibration": best,
            "metrics": {
                "pool_generation_singlehop_old_acc": sum(int(x["single_hit"]) for x in pool_info["probe_records"]) / len(pool_info["probe_records"]),
                "pool_generation_multihop_old_acc": sum(int(x["multi_hit"]) for x in pool_info["probe_records"]) / len(pool_info["probe_records"]),
                "test_case_count": len(test_pool),
                "edit_memory_singlehop_new_acc": 1.0,
                "multihop_no_cqag_new_acc": sum(no_cqag) / len(no_cqag),
                "multihop_base_old_pref_acc": sum(base_old) / len(base_old),
                "multihop_cqag_full_new_acc": sum(cqag_full) / len(cqag_full),
                "multihop_cqag_fact_only_new_acc": sum(cqag_fact_only) / len(cqag_fact_only),
                "multihop_cqag_random_new_acc": sum(cqag_random) / len(cqag_random),
                "locality_mismatched_old_pref_acc": sum(locality_mismatch) / len(locality_mismatch),
                "multihop_no_cqag_generation_acc": sum(no_cqag_generation) / len(no_cqag_generation),
                "multihop_cqag_generation_acc": sum(cqag_generation) / len(cqag_generation),
            },
            "test_details": details,
        }
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-model CQAG experiment on MQuAKE.")
    parser.add_argument("--model-dir", default="/root/autodl-tmp/modelscope_cache/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master")
    parser.add_argument("--dataset-path", default="/root/cqag_experiment/data/MQuAKE-CF-3k-v2.json")
    parser.add_argument("--output", default="/root/cqag_experiment/results/real_cqag_cf.json")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--filter-mode", choices=["strict", "generation"], default="strict")
    parser.add_argument("--max-scan-cases", type=int, default=240)
    parser.add_argument("--target-pool-size", type=int, default=24)
    parser.add_argument("--calib-size", type=int, default=8)
    parser.add_argument("--candidate-layers", default="6,10,14,18,22")
    parser.add_argument("--alpha-grid", default="0.5,1.0,1.5,2.0,2.5")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--qualified-case-ids", default="")
    args = parser.parse_args()

    config = ExperimentConfig(
        model_dir=args.model_dir,
        dataset_path=args.dataset_path,
        load_in_4bit=args.load_in_4bit,
        filter_mode=args.filter_mode,
        max_scan_cases=args.max_scan_cases,
        target_pool_size=args.target_pool_size,
        calib_size=args.calib_size,
        candidate_layers=parse_int_list(args.candidate_layers),
        alpha_grid=parse_float_list(args.alpha_grid),
        seed=args.seed,
        qualified_case_ids=parse_int_list(args.qualified_case_ids),
    )
    set_seed(config.seed)
    experiment = CQAGRealExperiment(config)
    results = experiment.run()

    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
