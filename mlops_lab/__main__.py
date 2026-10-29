"""Run offline model training and deployment gates; optionally write a report."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .core import experiment
from .report import render


def main():
    parser = argparse.ArgumentParser(description="Synthetic MLOps training + drift + canary assessment")
    parser.add_argument('--scenario', choices=['healthy','shifted','degraded'], default='healthy')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--json-out', type=Path)
    parser.add_argument('--html-out', type=Path)
    args = parser.parse_args()
    result = experiment(args.scenario, seed=args.seed)
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.json_out:
        args.json_out.write_text(payload, encoding='utf-8')
    if args.html_out:
        args.html_out.write_text(render(result), encoding='utf-8')
    print(json.dumps({'scenario':args.scenario,
                      'decision':result['canary_gate']['decision'],
                      'candidate_f1':round(result['candidate_metrics']['f1'], 3),
                      'feature_psi':{k:round(v,3) for k,v in result['feature_drift_psi'].items()},
                      'failing_checks':[c['check'] for c in result['canary_gate']['checks'] if not c['passed']]},indent=2))


if __name__=='__main__':
    main()
