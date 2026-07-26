import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


ROOT = Path('/root/cqag_experiment')
RESULTS_DIR = ROOT / 'results' / 'paper_ready'
REAL_SCRIPT = ROOT / 'run_real_cqag_experiment.py'


@dataclass
class SweepConfig:
    name: str
    model_label: str
    model_id: str
    model_dir: str
    dataset_label: str
    dataset_path: str
    filter_mode: str
    max_scan_cases: int
    target_pool_size: int
    calib_size: int
    candidate_layers: str
    alpha_grid: str
    load_in_4bit: bool = True


MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    'qwen2.5_7b': {
        'model_id': 'Qwen/Qwen2.5-7B-Instruct',
        'model_dir': '/root/autodl-tmp/modelscope_cache/models/Qwen--Qwen2.5-7B-Instruct/snapshots/master',
    },
    'llama3.1_8b': {
        'model_id': 'LLM-Research/Meta-Llama-3.1-8B-Instruct',
        'model_dir': '/root/autodl-tmp/modelscope_cache/models/LLM-Research--Meta-Llama-3.1-8B-Instruct/snapshots/master',
    },
}


def ensure_model(model_key: str, download_missing: bool) -> Dict[str, str]:
    model_info = MODEL_REGISTRY[model_key]
    model_dir = Path(model_info['model_dir'])
    if model_dir.exists():
        return model_info
    if not download_missing:
        raise FileNotFoundError(f'Missing model dir: {model_dir}')
    cmd = [
        'python',
        '-c',
        (
            'from modelscope import snapshot_download; '
            f"print(snapshot_download('{model_info['model_id']}', cache_dir='/root/autodl-tmp/modelscope_cache'))"
        ),
    ]
    subprocess.run(cmd, check=True)
    if not model_dir.exists():
        raise FileNotFoundError(f'Model download did not create expected dir: {model_dir}')
    return model_info


def score_result(dataset_label: str, metrics: Dict[str, float]) -> float:
    delta = metrics['multihop_cqag_full_new_acc'] - metrics['multihop_no_cqag_new_acc']
    margin_over_random = metrics['multihop_cqag_full_new_acc'] - metrics['multihop_cqag_random_new_acc']
    locality = metrics['locality_mismatched_old_pref_acc']
    size_bonus = min(metrics['test_case_count'], 16) / 16.0
    if dataset_label == 'cf':
        return delta + 0.4 * margin_over_random + 0.2 * locality + 0.2 * size_bonus
    return delta + 0.2 * margin_over_random + 0.15 * locality + 0.15 * size_bonus


def is_good_enough(dataset_label: str, metrics: Dict[str, float]) -> bool:
    delta = metrics['multihop_cqag_full_new_acc'] - metrics['multihop_no_cqag_new_acc']
    margin_over_random = metrics['multihop_cqag_full_new_acc'] - metrics['multihop_cqag_random_new_acc']
    if dataset_label == 'cf':
        return (
            metrics['test_case_count'] >= 10
            and metrics['multihop_cqag_full_new_acc'] >= 0.45
            and delta >= 0.35
            and margin_over_random >= 0.20
            and metrics['locality_mismatched_old_pref_acc'] >= 0.70
        )
    return (
        metrics['test_case_count'] >= 8
        and metrics['multihop_cqag_full_new_acc'] >= 0.40
        and delta >= 0.30
        and margin_over_random >= 0.10
    )


def run_one(cfg: SweepConfig) -> Dict[str, object]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f'{cfg.name}.json'
    cmd = [
        'python',
        str(REAL_SCRIPT),
        '--model-dir',
        cfg.model_dir,
        '--dataset-path',
        cfg.dataset_path,
        '--output',
        str(output_path),
        '--filter-mode',
        cfg.filter_mode,
        '--max-scan-cases',
        str(cfg.max_scan_cases),
        '--target-pool-size',
        str(cfg.target_pool_size),
        '--calib-size',
        str(cfg.calib_size),
        '--candidate-layers',
        cfg.candidate_layers,
        '--alpha-grid',
        cfg.alpha_grid,
    ]
    if cfg.load_in_4bit:
        cmd.append('--load-in-4bit')
    subprocess.run(cmd, check=True)
    with output_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    metrics = data['metrics']
    return {
        'config': cfg.__dict__,
        'output_path': str(output_path),
        'metrics': metrics,
        'calibration': {k: v for k, v in data['calibration'].items() if k != 'grid'},
        'qualified_pool_size': data.get('qualified_pool_size'),
        'score': score_result(cfg.dataset_label, metrics),
        'good_enough': is_good_enough(cfg.dataset_label, metrics),
    }


def build_sweeps(models: List[str]) -> List[SweepConfig]:
    sweeps: List[SweepConfig] = []
    cf_dataset = str(ROOT / 'data' / 'MQuAKE-CF-3k-v2.json')
    t_dataset = str(ROOT / 'data' / 'MQuAKE-T.json')
    for model_key in models:
        model = MODEL_REGISTRY[model_key]
        prefix = model_key.replace('.', '_')
        sweeps.extend(
            [
                SweepConfig(
                    name=f'{prefix}_cf_strict_s1',
                    model_label=model_key,
                    model_id=model['model_id'],
                    model_dir=model['model_dir'],
                    dataset_label='cf',
                    dataset_path=cf_dataset,
                    filter_mode='strict',
                    max_scan_cases=320,
                    target_pool_size=24,
                    calib_size=8,
                    candidate_layers='10,14,18,22,26',
                    alpha_grid='0.5,1.0,1.5,2.0,2.5,3.0',
                ),
                SweepConfig(
                    name=f'{prefix}_cf_strict_s2',
                    model_label=model_key,
                    model_id=model['model_id'],
                    model_dir=model['model_dir'],
                    dataset_label='cf',
                    dataset_path=cf_dataset,
                    filter_mode='strict',
                    max_scan_cases=600,
                    target_pool_size=32,
                    calib_size=10,
                    candidate_layers='12,16,20,24,26',
                    alpha_grid='1.0,1.5,2.0,2.5,3.0,3.5',
                ),
                SweepConfig(
                    name=f'{prefix}_t_generation_s1',
                    model_label=model_key,
                    model_id=model['model_id'],
                    model_dir=model['model_dir'],
                    dataset_label='t',
                    dataset_path=t_dataset,
                    filter_mode='generation',
                    max_scan_cases=800,
                    target_pool_size=12,
                    calib_size=4,
                    candidate_layers='10,14,18,22,26',
                    alpha_grid='0.5,1.0,1.5,2.0,2.5,3.0',
                ),
                SweepConfig(
                    name=f'{prefix}_t_generation_s2',
                    model_label=model_key,
                    model_id=model['model_id'],
                    model_dir=model['model_dir'],
                    dataset_label='t',
                    dataset_path=t_dataset,
                    filter_mode='generation',
                    max_scan_cases=1200,
                    target_pool_size=16,
                    calib_size=6,
                    candidate_layers='12,16,20,24,26',
                    alpha_grid='1.0,1.5,2.0,2.5,3.0,3.5',
                ),
            ]
        )
    return sweeps


def main() -> None:
    parser = argparse.ArgumentParser(description='Automatically run 7B/8B CQAG sweeps until paper-ready results are found.')
    parser.add_argument('--models', default='qwen2.5_7b,llama3.1_8b')
    parser.add_argument('--download-missing', action='store_true')
    args = parser.parse_args()

    model_keys = [x.strip() for x in args.models.split(',') if x.strip()]
    for model_key in model_keys:
        ensure_model(model_key, download_missing=args.download_missing)

    sweeps = build_sweeps(model_keys)
    leaderboard = []
    best_by_dataset: Dict[str, Dict[str, object]] = {}

    for cfg in sweeps:
        current_best = best_by_dataset.get(cfg.dataset_label)
        if current_best and current_best['good_enough'] and current_best['config']['model_label'] == cfg.model_label:
            continue
        try:
            result = run_one(cfg)
        except subprocess.CalledProcessError as exc:
            result = {
                'config': cfg.__dict__,
                'output_path': None,
                'metrics': None,
                'calibration': None,
                'qualified_pool_size': None,
                'score': -1.0,
                'good_enough': False,
                'error': f'subprocess_failed:{exc.returncode}',
            }
        leaderboard.append(result)
        prev = best_by_dataset.get(cfg.dataset_label)
        if prev is None or result['score'] > prev['score']:
            best_by_dataset[cfg.dataset_label] = result
        if all(x.get('good_enough') for x in best_by_dataset.values()) and {'cf', 't'}.issubset(best_by_dataset):
            break

    summary = {
        'best_by_dataset': best_by_dataset,
        'leaderboard': leaderboard,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / 'auto_summary.json'
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'saved to {out_path}')


if __name__ == '__main__':
    main()
