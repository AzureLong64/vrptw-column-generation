"""Command-line runs with machine-readable iteration and solution artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from .instance import read_instance
from .solver import Config, solve
from .pricing import enumerate_routes
from .master import Master


class Reporter:
    def __init__(self, folder: Path, quiet=False):
        folder.mkdir(parents=True, exist_ok=True)
        self.file = (folder/'events.jsonl').open('w',encoding='utf-8')
        self.quiet = quiet

    def __call__(self, event):
        self.file.write(json.dumps(event,ensure_ascii=False)+'\n')
        self.file.flush()
        if self.quiet:
            return
        if event['event']=='iteration' and (event['iteration']==1 or event['iteration']%10==0 or event['emp_certified']):
            print(f"[{event['algorithm']:8s}] it={event['iteration']:4d} round={event['outer_round']:3d} "
                  f"rows={event['rows']:4d} cols={event['columns']:5d} z={event['rmp_objective']:.6f} "
                  f"under={event['uncovered']:3d} +cols={event['new_columns']:3d} "
                  f"{event['pricing_kind']} {event['elapsed']:.2f}s",flush=True)
        elif event['event']=='separation':
            print(f"[{event['algorithm']:8s}] separate +{event['new_rows']} rows",flush=True)
        elif event['event']=='finish':
            print(f"[{event['algorithm']:8s}] status={event['status']} certified={event['certified']} "
                  f"z={event['objective']} time={event['elapsed']:.2f}s",flush=True)

    def close(self):
        self.file.close()


def save_result(folder: Path, result: dict):
    history = result.pop('history')
    (folder/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if history:
        with (folder/'iterations.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    with (folder/'active_routes.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f)
        writer.writerow(['column','value','cost','load','return_time','path'])
        for r in result['active_routes']:
            writer.writerow([r['column'],r['value'],r['cost'],r['load'],r['return_time'],
                             ' -> '.join(map(str,r['path']))])


def main(argv=None):
    parser=argparse.ArgumentParser(description='VRPTW ordinary CG and dyn-SAR, with exact elementary pricing certification')
    sub=parser.add_subparsers(dest='command',required=True)
    project=Path(__file__).resolve().parents[1]
    for name in ('solve','compare','demo','validate'):
        p=sub.add_parser(name)
        p.add_argument('--instance',type=Path,default=project/'examples'/'tiny6.txt')
        p.add_argument('--limit-customers',type=int,help='Use the first N customers; explicitly creates a smaller instance')
        p.add_argument('--output',type=Path,default=project/'results'/name)
        p.add_argument('--time-limit',type=float,default=120,help='Seconds per algorithm, including LP and pricing')
        p.add_argument('--pricing-time-limit',type=float,default=20)
        p.add_argument('--max-labels',type=int,default=300000)
        p.add_argument('--max-iterations',type=int,default=10000)
        p.add_argument('--max-columns',type=int,default=300)
        p.add_argument('--kappa',type=int,default=20)
        p.add_argument('--no-heuristics',action='store_true')
        p.add_argument('--quiet',action='store_true')
        if name=='solve':
            p.add_argument('--algorithm',choices=['standard','dyn-sar'],default='standard')
    args=parser.parse_args(argv)
    instance=read_instance(args.instance,args.limit_customers)
    config=Config(time_limit=args.time_limit,pricing_time_limit=args.pricing_time_limit,
                  max_labels=args.max_labels,max_iterations=args.max_iterations,max_columns=args.max_columns,
                  kappa=args.kappa,heuristics=not args.no_heuristics)
    algorithms=[args.algorithm] if args.command=='solve' else ['standard','dyn-sar']
    if args.command in ('validate','demo') and instance.n>10:
        parser.error('demo/validate use exhaustive reference and require <=10 customers; use compare for larger instances')
    results=[]
    for algorithm in algorithms:
        folder=args.output/algorithm
        reporter=Reporter(folder,args.quiet)
        try:
            result=solve(instance,algorithm,config,reporter)
        finally:
            reporter.close()
        save_result(folder,result)
        results.append(result)
    summary={"instance":instance.name,"n_customers":instance.n,"model":"elementary-route covering LP, no fleet constraint",
             "results":[{k:r[k] for k in ('algorithm','status','certified_optimal','lp_optimum','last_rmp_objective',
                                          'iterations','outer_rounds','generated_columns','generated_rows','elapsed_seconds',
                                          'certified_emp_lower_bound','feasible_lp_upper_bound')} for r in results]}
    if args.command in ('validate','demo'):
        all_routes=enumerate_routes(instance)
        reference=Master(instance,[{i} for i in range(1,instance.n+1)])
        reference.add_routes(all_routes)
        optimum=reference.solve().objective
        passed=all(r['certified_optimal'] and abs(r['lp_optimum']-optimum)<=1e-6 for r in results)
        summary.update(reference_lp=optimum,enumerated_customer_sets=len(all_routes),reference_check_passed=passed)
        print(f"EXHAUSTIVE REFERENCE: z={optimum:.9f}; both algorithms match={passed}")
    elif len(results)==2 and all(r['certified_optimal'] for r in results):
        summary['objectives_match']=abs(results[0]['lp_optimum']-results[1]['lp_optimum'])<=1e-6
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"Results: {args.output.resolve()}")
    if summary.get('reference_check_passed') is False or summary.get('objectives_match') is False:
        return 1
    return 0 if all(r['certified_optimal'] for r in results) else 2


if __name__=='__main__':
    sys.exit(main())
