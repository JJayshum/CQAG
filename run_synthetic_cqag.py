import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    seed: int = 7
    num_countries: int = 128
    hidden_size: int = 64
    train_steps: int = 2500
    train_lr: float = 3e-3
    batch_size: int = 64
    edit_count: int = 32
    val_edit_count: int = 16
    edit_steps: int = 400
    edit_lr: float = 8e-2
    edit_reg: float = 5e-4
    alpha_grid: tuple = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TwoHopReasoner(nn.Module):
    def __init__(self, num_countries: int, num_leaders: int, num_spouses: int, hidden_size: int):
        super().__init__()
        self.country_emb = nn.Embedding(num_countries, hidden_size)
        self.bridge = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.leader_head = nn.Linear(hidden_size, num_leaders)
        self.spouse_head = nn.Linear(hidden_size, num_spouses)

    def encode_bridge(self, country_ids: torch.Tensor) -> torch.Tensor:
        return self.bridge(self.country_emb(country_ids))

    def forward(
        self,
        country_ids: torch.Tensor,
        task: str,
        bridge_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bridge = self.encode_bridge(country_ids)
        if bridge_delta is not None:
            bridge = bridge + bridge_delta
        if task == "leader":
            return self.leader_head(bridge)
        if task == "spouse":
            return self.spouse_head(bridge)
        raise ValueError(f"Unknown task: {task}")


def build_edit_memory(world: Dict[str, torch.Tensor]) -> Dict[int, int]:
    return {
        int(country.item()): int(world["leaders_new"][country].item())
        for country in world["edit_countries"]
    }


def build_world(config: Config) -> Dict[str, torch.Tensor]:
    n = config.num_countries
    countries = torch.arange(n, dtype=torch.long)
    leaders_old = torch.randperm(n, dtype=torch.long)
    spouses = torch.randperm(n, dtype=torch.long)
    spouses_by_country_old = spouses[leaders_old]

    edit_countries = torch.randperm(n, dtype=torch.long)[: config.edit_count]
    shift = torch.roll(edit_countries, shifts=-1)

    leaders_new = leaders_old.clone()
    leaders_new[edit_countries] = leaders_old[shift]
    spouses_by_country_new = spouses[leaders_new]

    val_edit_countries = edit_countries[: config.val_edit_count]
    test_edit_countries = edit_countries[config.val_edit_count :]
    mask = torch.ones(n, dtype=torch.bool)
    mask[edit_countries] = False
    locality_countries = countries[mask]

    return {
        "countries": countries,
        "leaders_old": leaders_old,
        "leaders_new": leaders_new,
        "spouses": spouses,
        "spouses_by_country_old": spouses_by_country_old,
        "spouses_by_country_new": spouses_by_country_new,
        "edit_countries": edit_countries,
        "val_edit_countries": val_edit_countries,
        "test_edit_countries": test_edit_countries,
        "locality_countries": locality_countries,
    }


def train_base_model(model: TwoHopReasoner, world: Dict[str, torch.Tensor], config: Config, device: torch.device) -> List[Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train_lr)
    countries = world["countries"].to(device)
    leaders_old = world["leaders_old"].to(device)
    spouses_old = world["spouses_by_country_old"].to(device)
    history: List[Dict[str, float]] = []

    for step in range(1, config.train_steps + 1):
        batch_idx = torch.randint(0, len(countries), (config.batch_size,), device=device)
        batch_countries = countries[batch_idx]
        leader_targets = leaders_old[batch_idx]
        spouse_targets = spouses_old[batch_idx]

        leader_logits = model(batch_countries, "leader")
        spouse_logits = model(batch_countries, "spouse")
        loss = F.cross_entropy(leader_logits, leader_targets) + F.cross_entropy(spouse_logits, spouse_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 250 == 0 or step == 1:
            with torch.no_grad():
                leader_acc = accuracy(model, countries, leaders_old, "leader", device)
                spouse_acc = accuracy(model, countries, spouses_old, "spouse", device)
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "leader_acc": float(leader_acc),
                    "spouse_acc": float(spouse_acc),
                }
            )
            print(
                f"[train] step={step:04d} loss={loss.item():.4f} "
                f"leader_acc={leader_acc:.3f} spouse_acc={spouse_acc:.3f}"
            )
    return history


def accuracy(
    model: TwoHopReasoner,
    countries: torch.Tensor,
    targets: torch.Tensor,
    task: str,
    device: torch.device,
    bridge_delta: torch.Tensor | None = None,
    edit_memory: Dict[int, int] | None = None,
) -> float:
    countries = countries.to(device)
    targets = targets.to(device)
    logits = model(countries, task, bridge_delta=bridge_delta)
    if task == "leader" and edit_memory:
        for row, country in enumerate(countries.detach().cpu().tolist()):
            if country in edit_memory:
                logits[row].fill_(-1e9)
                logits[row, edit_memory[country]] = 1e9
    preds = logits.argmax(dim=-1)
    return float((preds == targets).float().mean().item())


def build_aligned_delta(
    model: TwoHopReasoner,
    world: Dict[str, torch.Tensor],
    countries: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        old_bridge = model.encode_bridge(countries.to(device))
        new_country_sources = torch.empty_like(countries)
        old_leaders = world["leaders_old"]
        new_leaders = world["leaders_new"]
        leader_to_country = {int(leader.item()): idx for idx, leader in enumerate(old_leaders)}
        for idx, country in enumerate(countries.tolist()):
            new_leader = int(new_leaders[country].item())
            new_country_sources[idx] = leader_to_country[new_leader]
        new_bridge = model.encode_bridge(new_country_sources.to(device))
    return new_bridge - old_bridge


def shuffled_delta(delta: torch.Tensor, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=delta.device)
    gen.manual_seed(seed)
    perm = torch.randperm(delta.shape[0], generator=gen, device=delta.device)
    return delta[perm]


def choose_alpha(
    model: TwoHopReasoner,
    world: Dict[str, torch.Tensor],
    config: Config,
    device: torch.device,
) -> Dict[str, float]:
    val_countries = world["val_edit_countries"]
    val_targets = world["spouses_by_country_new"][val_countries]
    delta = build_aligned_delta(model, world, val_countries, device)

    best_alpha = 0.0
    best_acc = -1.0
    scores: Dict[str, float] = {}
    for alpha in config.alpha_grid:
        acc = accuracy(model, val_countries, val_targets, "spouse", device, bridge_delta=alpha * delta)
        scores[str(alpha)] = acc
        if acc > best_acc:
            best_alpha = float(alpha)
            best_acc = acc
    return {"best_alpha": best_alpha, "best_acc": best_acc, "scores": scores}


def summarize_predictions(
    model: TwoHopReasoner,
    world: Dict[str, torch.Tensor],
    countries: torch.Tensor,
    task: str,
    targets: torch.Tensor,
    device: torch.device,
    bridge_delta: torch.Tensor | None = None,
    edit_memory: Dict[int, int] | None = None,
) -> Dict[str, float]:
    return {
        "accuracy": accuracy(
            model,
            countries,
            targets,
            task,
            device,
            bridge_delta=bridge_delta,
            edit_memory=edit_memory,
        ),
        "count": int(len(countries)),
    }


def run_experiment(config: Config, output_dir: Path) -> Dict[str, object]:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    world = build_world(config)
    base_model = TwoHopReasoner(
        num_countries=config.num_countries,
        num_leaders=config.num_countries,
        num_spouses=config.num_countries,
        hidden_size=config.hidden_size,
    ).to(device)

    train_history = train_base_model(base_model, world, config, device)
    edit_memory = build_edit_memory(world)

    best_alpha_info = choose_alpha(base_model, world, config, device)
    alpha = best_alpha_info["best_alpha"]

    test_edit_countries = world["test_edit_countries"]
    locality_countries = world["locality_countries"]
    aligned_delta = build_aligned_delta(base_model, world, test_edit_countries, device)
    random_delta = shuffled_delta(aligned_delta, seed=config.seed + 13)

    results = {
        "config": asdict(config),
        "best_alpha_info": best_alpha_info,
        "base_pre_edit": {
            "single_hop_edit_old_target": summarize_predictions(
                base_model,
                world,
                test_edit_countries,
                "leader",
                world["leaders_old"][test_edit_countries],
                device,
            ),
            "multi_hop_edit_old_target": summarize_predictions(
                base_model,
                world,
                test_edit_countries,
                "spouse",
                world["spouses_by_country_old"][test_edit_countries],
                device,
            ),
        },
        "post_edit_no_cqag": {
            "single_hop_edit_new_target": summarize_predictions(
                base_model,
                world,
                test_edit_countries,
                "leader",
                world["leaders_new"][test_edit_countries],
                device,
                edit_memory=edit_memory,
            ),
            "single_hop_locality_old_target": summarize_predictions(
                base_model,
                world,
                locality_countries,
                "leader",
                world["leaders_old"][locality_countries],
                device,
                edit_memory=edit_memory,
            ),
            "multi_hop_edit_new_target": summarize_predictions(
                base_model,
                world,
                test_edit_countries,
                "spouse",
                world["spouses_by_country_new"][test_edit_countries],
                device,
            ),
            "multi_hop_edit_old_target": summarize_predictions(
                base_model,
                world,
                test_edit_countries,
                "spouse",
                world["spouses_by_country_old"][test_edit_countries],
                device,
            ),
            "multi_hop_locality_old_target": summarize_predictions(
                base_model,
                world,
                locality_countries,
                "spouse",
                world["spouses_by_country_old"][locality_countries],
                device,
            ),
        },
        "post_edit_cqag": {
            "alpha": alpha,
            "multi_hop_edit_new_target": summarize_predictions(
                base_model,
                world,
                test_edit_countries,
                "spouse",
                world["spouses_by_country_new"][test_edit_countries],
                device,
                bridge_delta=alpha * aligned_delta,
            ),
            "multi_hop_edit_old_target": summarize_predictions(
                base_model,
                world,
                test_edit_countries,
                "spouse",
                world["spouses_by_country_old"][test_edit_countries],
                device,
                bridge_delta=alpha * aligned_delta,
            ),
            "multi_hop_random_vector": summarize_predictions(
                base_model,
                world,
                test_edit_countries,
                "spouse",
                world["spouses_by_country_new"][test_edit_countries],
                device,
                bridge_delta=alpha * random_delta,
            ),
            "multi_hop_locality_old_target": summarize_predictions(
                base_model,
                world,
                locality_countries,
                "spouse",
                world["spouses_by_country_old"][locality_countries],
                device,
            ),
        },
        "train_history": train_history,
        "edit_memory_size": len(edit_memory),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "summary.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {result_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic CQAG experiment.")
    parser.add_argument("--output-dir", type=Path, default=Path("/root/cqag_experiment/results"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = Config(seed=args.seed)
    run_experiment(config, args.output_dir)


if __name__ == "__main__":
    main()
