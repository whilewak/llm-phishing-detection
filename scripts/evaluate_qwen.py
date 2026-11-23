#!/usr/bin/env python3
"""Run Phase 4 zero/few-shot evaluations on the unchanged frozen LLM splits.

HF_HOME=data/models/huggingface python scripts/evaluate_qwen.py
Use --mode zero or --mode few to resume just one variant. Cache hits preserve
original inference latency; replay wall throughput is explicitly marked.
"""
from __future__ import annotations
import argparse
import json
import statistics
import tomllib
from pathlib import Path

from phishbench.data.splits import load_split
from phishbench.detectors import load_detector
from phishbench.eval import evaluate, write_result

ROOT = Path(__file__).resolve().parents[1]


class Progress:
    def __init__(self, detector, n_validation):
        self.detector = detector
        self.name = detector.name
        self.count = 0
        self.n_validation = n_validation
    def fit(self, train, val):
        return self.detector.fit(train, val)
    def predict(self, emails):
        # The harness warms up after validation; bypass response cache so all
        # warmup calls actually execute the GPU and don't populate test cache.
        cache = self.detector.cache_dir
        if self.n_validation <= self.count < self.n_validation + 3:
            self.detector.cache_dir = None
        try:
            out = self.detector.predict(emails)
        finally:
            self.detector.cache_dir = cache
        self.count += len(emails)
        if self.count % 100 == 0:
            print(f"{self.name}: {self.count} predictions (validation + warmup + test)", flush=True)
        return out


def summarize(result):
    preds = result['predictions']
    details = [p['details'] for p in preds if p['error'] is None]
    hits = sum(d.get('cache_hit', False) for d in details)
    result['qwen'] = {
        'scoring': 'P(A | A or B); A = phishing, B = legitimate; float32 log-softmax',
        'thinking': False, 'temperature': 0, 'n_shots': result['config']['detector']['params']['n_shots'],
        'example_ids': details[0]['example_ids'] if details else [],
        'truncation_share': sum(d['truncated'] for d in details) / len(details) if details else None,
        'mean_answer_probability_mass': statistics.fmean(d['answer_probability_mass'] for d in details) if details else None,
        'cache_hits': hits,
        'validation_note': 'Frozen val_llm retained (1,382 negatives; ~14 false positives at 1%); threshold uncertainty remains.',
        'preliminary': True,
        'timing_note': 'Local machine timings may reflect memory pressure and idle sleep; do not treat them as controlled hardware comparisons.',
    }
    import mlx.core as mx
    result['environment']['mlx_device'] = mx.device_info()
    result['resources']['gpu'] = {'mlx_peak_memory_mb': mx.get_peak_memory() / 2**20}
    result['resources']['gpu_memory_basis'] = 'process lifetime MLX peak; variants run sequentially'
    secondary = result.get('detection', {}).get('secondary_recall_at_fpr')
    if secondary is not None:
        secondary['reportable'] = False
        secondary['note'] = '0.1% FPR is not supported by the 1,382-negative LLM subsample; retained only as raw harness output.'
    if hits:
        result['speed']['throughput_basis'] = 'cache replay wall time; not inference throughput'
    else:
        result['speed']['throughput_basis'] = 'uncached inference wall time with shared prompt KV cache'
    result['speed']['latency_basis'] = 'original uncached per-email latency, after shared prompt prefill warmup'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['zero', 'few', 'both'], default='both')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    split_dir = ROOT / 'data/splits/champa2023-v1'
    manifest = json.loads((split_dir / 'manifest.json').read_text())
    print('Loading and verifying frozen splits', flush=True)
    train = load_split(split_dir, 'train')
    val = load_split(split_dir, 'val_llm')
    test = load_split(split_dir, 'test_llm')
    for mode in (['zero', 'few'] if args.mode == 'both' else [args.mode]):
        path = ROOT / f'configs/models/qwen-{mode}.toml'
        config = tomllib.loads(path.read_text())
        params = config['detector']['params']
        detector = load_detector(config['detector']['class'], params)
        # Fail early on runtime/model setup, rather than recording 5,000 setup errors.
        detector._backend()
        config['runtime'] = detector.runtime
        config['data_ref'] = {'version': manifest['version'], 'splits': str(split_dir),
                              'train': 'train' if params['n_shots'] else None,
                              'val': 'val_llm', 'test': 'test_llm'}
        result = evaluate(Progress(detector, len(val)), test, train=train if params['n_shots'] else [],
            val=val, warmup=3, batch_size=1, n_bootstrap=args.bootstrap, config=config,
            source_groups={'true_phishing': manifest['true_phishing_sources']})
        summarize(result)
        output = write_result(result, ROOT / 'results')
        print(f"Wrote {output}; errors={result['errors']['count']}", flush=True)
        del result, detector
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
