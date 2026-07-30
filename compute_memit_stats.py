import argparse
import glob
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Compute official-style MEMIT mom2 statistics for Qwen.")
    parser.add_argument("--easyedit-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--stats-dir", required=True)
    parser.add_argument("--layers", default="4,5,6,7,8")
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--batch-tokens", type=int, default=12288)
    parser.add_argument(
        "--datasets-cache-dir",
        help="Optional Hugging Face datasets cache directory, useful when the root disk is constrained.",
    )
    parser.add_argument(
        "--local-wikipedia-glob",
        help="Glob for a local mirror of wikimedia/wikipedia 20231101.en parquet shards.",
    )
    args = parser.parse_args()

    if args.datasets_cache_dir:
        os.environ["HF_DATASETS_CACHE"] = args.datasets_cache_dir

    import sys

    sys.path.insert(0, args.easyedit_dir)
    # EasyEdit's top-level __init__ re-exports trainer.models as
    # ``easyeditor.models``. Remove that alias so Python resolves the
    # actual easyeditor/models package containing the official ROME stats.
    import easyeditor
    sys.modules.pop("easyeditor.models", None)
    if hasattr(easyeditor, "models"):
        delattr(easyeditor, "models")
    import importlib
    stats_module = importlib.import_module("easyeditor.models.rome.layer_stats")
    from datasets import load_dataset

    if args.local_wikipedia_glob:
        shard_paths = sorted(glob.glob(args.local_wikipedia_glob))
        if not shard_paths:
            raise RuntimeError(f"No Wikipedia shards match {args.local_wikipedia_glob}")

        def load_local_wikipedia(_dataset_name, _dataset_config):
            return load_dataset("parquet", data_files={"train": shard_paths})

        stats_module.load_dataset = load_local_wikipedia

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).eval().to("cuda:0")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for layer in (int(value) for value in args.layers.split(",") if value.strip()):
        module_name = f"model.layers.{layer}.mlp.down_proj"
        print(f"[mom2] layer={layer} module={module_name}", flush=True)
        stats_module.layer_stats(
            model,
            tokenizer,
            module_name,
            args.stats_dir,
            "wikipedia",
            ["mom2"],
            sample_size=args.sample_size,
            precision="float32",
            batch_tokens=args.batch_tokens,
            download=True,
            hparams=type("StatsConfig", (), {"device": 0})(),
        )


if __name__ == "__main__":
    main()
