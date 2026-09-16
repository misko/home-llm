from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description='Prepare, execute and report the controlled Qwen comparison.')
    parser.add_argument('root',type=Path)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare')
    sub.add_parser('status')
    sub.add_parser('prepare-nll')
    sub.add_parser('probe-sharing')
    sub.add_parser('supervise')
    sub.add_parser('performance')
    build=sub.add_parser('build-artifact');build.add_argument('arm');build.add_argument('input_model',type=Path);build.add_argument('--parent')
    nll=sub.add_parser('nll');nll.add_argument('arm')
    quant=sub.add_parser('check-quantization');quant.add_argument('arm')
    report=sub.add_parser('report');report.add_argument('--partition',choices=['development','final'],default='final')
    publish=sub.add_parser('publish');publish.add_argument('--partition',choices=['development','final'],default='final')
    run=sub.add_parser('run');run.add_argument('arm',choices=['qwen','qwen-ft-fineweb','qwen-heretic','qwen-heretic-plus'])
    run.add_argument('--partition',choices=['development','final'],default='final')
    run.add_argument('--limit',type=int);run.add_argument('--retry-infrastructure',action='store_true')
    code=sub.add_parser('score-code');code.add_argument('arm');code.add_argument('--partition',default='final')
    review=sub.add_parser('review-packet');review.add_argument('--partition',default='final')
    audit=sub.add_parser('audit-packet');audit.add_argument('--partition',default='final');audit.add_argument('--fraction',type=float,default=.2)
    ingest=sub.add_parser('import-reviews');ingest.add_argument('source',type=Path);ingest.add_argument('--partition',default='final')
    audit_ingest=sub.add_parser('import-audit');audit_ingest.add_argument('source',type=Path);audit_ingest.add_argument('--partition',default='final')
    args=parser.parse_args()
    if args.command=='prepare':
        from .prepare import prepare
        result=prepare(args.root)
    elif args.command=='performance':
        from .performance import run
        result=run(args.root)
    elif args.command=='supervise':
        from .supervise import supervise
        supervise(args.root);result={'supervisor':'returned'}
    elif args.command=='build-artifact':
        from .artifacts import build
        result=build(args.root,args.arm,args.input_model,parent=args.parent)
    elif args.command=='probe-sharing':
        from .sharing_probe import check
        result=check(args.root)
    elif args.command=='prepare-nll':
        from .nll import prepare
        result=prepare(args.root)
    elif args.command=='check-quantization':
        from .quantization import check
        result=check(args.root,args.arm)
    elif args.command=='nll':
        from .nll import run
        result=run(args.root,args.arm)
    elif args.command=='report':
        from .report import render
        result=str(render(args.root,partition=args.partition))
    elif args.command=='publish':
        from .report import publish
        result=publish(args.root,partition=args.partition)
    elif args.command=='run':
        from .execute import evaluate_arm
        result=evaluate_arm(args.root,args.arm,partition=args.partition,limit=args.limit,retry_infrastructure=args.retry_infrastructure)
    elif args.command=='score-code':
        from .code_eval import score_saved
        config=json.loads((args.root/'execution.json').read_text())
        validation=json.loads((args.root/'code-evaluator-validation.json').read_text())
        if not validation['complete'] or not validation['passed'] or validation['image']!=config['code_image']:
            raise RuntimeError('Coding evaluator has not passed canonical validation with this image')
        result=score_saved(args.root,args.arm,args.partition,config['code_image'])
    elif args.command=='review-packet':
        from .judgments import packet
        result=str(packet(args.root,args.partition))
    elif args.command=='audit-packet':
        from .judgments import audit_packet
        result=str(audit_packet(args.root,args.partition,fraction=args.fraction))
    elif args.command=='import-reviews':
        from .judgments import import_reviews
        import_reviews(args.root,args.partition,args.source);result={'imported':str(args.source)}
    elif args.command=='import-audit':
        from .judgments import import_audit_reviews
        result=str(import_audit_reviews(args.root,args.partition,args.source))
    else:
        result={name:json.loads((args.root/name).read_text()) for name in ['status.json','dataset-lock.json','artifacts.json'] if (args.root/name).exists()}
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
