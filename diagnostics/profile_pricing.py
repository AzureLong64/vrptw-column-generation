"""Attribute pricing time without changing either solver's search policy.

Run from the project root: python diagnostics/profile_pricing.py
Runs both algorithms with the default pricing pipeline, then with heuristics
disabled. Each algorithm gets the same limits within each experiment.
"""
from collections import Counter
from functools import wraps
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vrptw_cg import Config, read_instance, solve
from vrptw_cg.pricing import Pricer


def main():
    folder = ROOT / 'results' / 'diagnosis_r1_200'
    folder.mkdir(parents=True, exist_ok=True)
    instance = read_instance(ROOT.parent / 'instances-ijoc-dyn-sar-main' /
                             'VRPTW/homberger_200_customer_instances/R1_2_1.TXT')
    records = []
    originals = {name: getattr(Pricer, name) for name in ('greedy', 'labels', 'exact')}

    def instrument(name):
        @wraps(originals[name])
        def wrapped(self, *args, **kwargs):
            result = originals[name](self, *args, **kwargs)
            # Complete exact() already includes its internal labels() calls.
            if name != 'labels' or kwargs.get('network_size') is not None:
                records.append({
                    'stage': result.kind, 'seconds': result.seconds,
                    'routes': len(result.routes), 'labels': result.labels,
                    'reason': result.reason, 'complete': result.complete,
                    'refinements': result.refinements,
                })
            return result
        return wrapped

    for name in originals:
        setattr(Pricer, name, instrument(name))
    summaries = []
    try:
        for heuristic in (True, False):
            for algorithm in ('standard', 'dyn-sar'):
                records.clear()
                tag = f'{algorithm}_{"default" if heuristic else "no_heuristics"}'
                print(f'START {tag}', flush=True)
                def progress(event):
                    if event['event'] == 'iteration' and event['iteration'] % 40 == 0:
                        print(f"  {tag}: it={event['iteration']} elapsed={event['elapsed']:.2f}s", flush=True)
                result = solve(instance, algorithm, Config(time_limit=180, pricing_time_limit=30,
                               heuristics=heuristic), callback=progress)
                stages = {}
                for stage in sorted({r['stage'] for r in records}):
                    rows = [r for r in records if r['stage'] == stage]
                    stages[stage] = {
                        'calls': len(rows), 'seconds': sum(r['seconds'] for r in rows),
                        'successful_calls': sum(bool(r['routes']) for r in rows),
                        'routes_returned': sum(r['routes'] for r in rows),
                        'labels': sum(r['labels'] for r in rows),
                        'reasons': dict(Counter(r['reason'] for r in rows)),
                        'empty_call_seconds': sum(r['seconds'] for r in rows if not r['routes']),
                    }
                summary = {k: result[k] for k in ('algorithm', 'status', 'lp_optimum',
                           'elapsed_seconds', 'lp_seconds', 'pricing_seconds', 'iterations',
                           'outer_rounds', 'generated_columns', 'generated_rows')}
                summary.update(heuristics=heuristic, stages=stages)
                summaries.append(summary)
                artifact = {'summary': summary, 'pricing_calls': list(records), 'result': result}
                (folder / f'{tag}.json').write_text(json.dumps(artifact, indent=2), encoding='utf-8')
                (folder / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
                print(json.dumps(summary), flush=True)
    finally:
        for name, original in originals.items():
            setattr(Pricer, name, original)


if __name__ == '__main__':
    main()
