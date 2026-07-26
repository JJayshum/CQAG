import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_DIR = "/root/autodl-tmp/modelscope_cache/models/Qwen--Qwen2.5-0.5B-Instruct/snapshots/master"
DATA_PATH = "/root/cqag_experiment/data/MQuAKE-CF-3k-v2.json"


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[\s\n\t]+", " ", text)
    text = re.sub(r"[\"'`.,:;!?()\[\]{}]", "", text)
    return text


def matches(pred: str, gold: str, aliases: list[str] | None = None) -> bool:
    candidates = [gold] + (aliases or [])
    p = normalize(pred)
    for c in candidates:
        c_norm = normalize(c)
        if c_norm and (p == c_norm or c_norm in p):
            return True
    return False


def generate(tokenizer, model, question: str, max_new_tokens: int = 24) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question + " Answer with just the answer."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def main() -> None:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    sample_size = 40
    results = []
    old_single_ok = 0
    old_multi_ok = 0
    both_ok = 0

    for item in data[:sample_size]:
        single = item["single_hops"][0]
        single_pred = generate(tokenizer, model, single["question"])
        multi_pred = generate(tokenizer, model, item["questions"][0])
        single_hit = matches(single_pred, single["answer"], single.get("answer_alias", []))
        multi_hit = matches(multi_pred, item["answer"], item.get("answer_alias", []))
        old_single_ok += int(single_hit)
        old_multi_ok += int(multi_hit)
        both_ok += int(single_hit and multi_hit)
        results.append(
            {
                "case_id": item["case_id"],
                "single_q": single["question"],
                "single_gold": single["answer"],
                "single_pred": single_pred,
                "single_hit": single_hit,
                "multi_q": item["questions"][0],
                "multi_gold": item["answer"],
                "multi_pred": multi_pred,
                "multi_hit": multi_hit,
                "new_single_target": item["new_single_hops"][0]["answer"],
                "new_multi_target": item["new_answer"],
            }
        )
        print(
            f"case={item['case_id']} single_hit={single_hit} multi_hit={multi_hit} "
            f"single_pred={single_pred[:40]!r} multi_pred={multi_pred[:40]!r}"
        )

    out = {
        "sample_size": sample_size,
        "old_single_acc": old_single_ok / sample_size,
        "old_multi_acc": old_multi_ok / sample_size,
        "both_old_acc": both_ok / sample_size,
        "details": results,
    }
    out_path = Path("/root/cqag_experiment/results/real_model_probe.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "details"}, ensure_ascii=False, indent=2))
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
