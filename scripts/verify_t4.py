"""Independent T4 acceptance without another Test generation pass."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
from t4 import RUN, FREEZE, contract


def main():
    frozen=json.loads(FREEZE.read_text())
    if contract()!=frozen:raise ValueError('T4 frozen environment changed')
    for arm,expected_tensors in [('baseline',0),('adapter',16)]:
        path=RUN/arm
        result=json.loads((path/'run_result.json').read_text())
        if result['adapter_tensors_loaded']!=expected_tensors or not result['independent_process']:
            raise ValueError('incorrect loaded model: '+arm)
        if len((path/'predictions.jsonl').read_text().splitlines())!=190:
            raise ValueError('incomplete first-pass predictions: '+arm)
    checks={}
    commands=[('pytest',[sys.executable,'-m','pytest','-q']),
        ('frozen_T2',[sys.executable,'scripts/prepare_balanced.py','--validate']),
        ('recompute',[sys.executable,'scripts/recompute_test.py'])]
    for name,command in commands:
        run=subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT/'src')),
            capture_output=True,text=True)
        (RUN/f'final-{name}.log').write_text(run.stdout+run.stderr)
        if run.returncode:raise RuntimeError(name+' failed: '+run.stdout+run.stderr)
        checks[name]={'exit_code':run.returncode,'stdout':run.stdout.strip() if name=='pytest' else 'saved in final-'+name+'.log'}
    comparison=json.loads((RUN/'recomputed_comparison.json').read_text())
    if not comparison['all_metrics_recomputed_match'] or not comparison['all_prompt_tokens_identical']:
        raise ValueError('T4 fairness audit failed')
    report={'status':'passed','experiment_id':frozen['experiment_id'],
        'test_predictions_per_arm':190,'base_adapter_tensors':0,'selected_adapter_tensors':16,
        'all_prompts_identical':True,'all_metrics_recomputed_match':True,
        'selected_before_test':True,'checks':checks,'time_unix':time.time()}
    (RUN/'final_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
