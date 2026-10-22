import json
import os
from pathlib import Path
import time

from .metrics import aggregate, parse_prediction, score_prediction


def generation_input(record):
    if not record['messages'] or record['messages'][-1]['role']!='assistant':
        raise ValueError('last message must be assistant target')
    return {'tools':record['tools'],'messages':record['messages'][:-1]}


def read_completed(path, contract, sample_ids):
    path=Path(path)
    if not path.exists():return {}
    if not (path/'contract.json').is_file() or json.loads((path/'contract.json').read_text())!=contract:
        raise ValueError('evaluation contract mismatch')
    completed={};allowed=set(sample_ids)
    predictions=path/'predictions.jsonl'
    for line in predictions.read_text().splitlines() if predictions.exists() else []:
        row=json.loads(line);sid=row['sample_id']
        if sid in completed:raise ValueError('duplicate prediction ID')
        if sid not in allowed:raise ValueError('unknown prediction ID')
        completed[sid]=row
    return completed


def recompute(records, predictions):
    if set(predictions)!={r['sample_id'] for r in records}:raise ValueError('incomplete predictions')
    scores=[]
    for e in records:
        p=predictions[e['sample_id']]
        parsed=parse_prediction(p['output'],p['stop_reason'])
        score=score_prediction(e,parsed)
        scores.append(dict(sample_id=e['sample_id'],source_record_id=e['source_record_id'],
                           group_id=e['group_id'],prediction=parsed,**score))
    return scores,aggregate(scores)


def run_evaluation(model,tokenizer,records,path,contract,check_resources=lambda:None):
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    path=Path(path);ids=[e['sample_id'] for e in records]
    if len(ids)!=len(set(ids)):raise ValueError('duplicate input IDs')
    completed=read_completed(path,contract,ids)
    if not path.exists():
        path.mkdir(parents=True)
        (path/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    model.eval();start=time.perf_counter()
    with (path/'predictions.jsonl').open('a') as log:
        for i,e in enumerate(records):
            if e['sample_id'] in completed:continue
            check_resources()
            inputs=generation_input(e)
            prompt=tokenizer.apply_chat_template(inputs['messages'],tools=inputs['tools'] or None,
                add_generation_prompt=True,return_dict=False)
            output=[];reason='length';tic=time.perf_counter();engine_error=None
            try:
                generator=generate_step(mx.array(prompt),model,
                    max_tokens=contract['decode']['max_tokens'],sampler=make_sampler(temp=0),prefill_step_size=256)
                try:
                    for token,_ in generator:
                        if token==tokenizer.eos_token_id:reason='eos';break
                        output.append(token)
                finally:generator.close()
            except RuntimeError as exc:
                reason='engine_error';engine_error=str(exc)
            row={'sample_id':e['sample_id'],'input':inputs,'prompt_ids':prompt,
                'output_ids':output,'output':tokenizer.decode(output),'stop_reason':reason,
                'seconds':time.perf_counter()-tic,'cumulative_mlx_peak_bytes':mx.get_peak_memory()}
            if engine_error:row['engine_error']=engine_error
            log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush();os.fsync(log.fileno())
            completed[e['sample_id']]=row
            mx.clear_cache()
            if engine_error:raise RuntimeError(engine_error)
            if (i+1)%10==0 or i+1==len(records):
                print('evaluation',path.name,i+1,'/',len(records),'elapsed',round(time.perf_counter()-start,1),flush=True)
    scores,summary=recompute(records,completed)
    summary['resource']={'output_tokens':sum(len(p['output_ids']) for p in completed.values()),
        'generation_seconds':sum(p['seconds'] for p in completed.values()),
        'length_stops':sum(p['stop_reason']=='length' for p in completed.values())}
    (path/'scores.jsonl').write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in scores))
    (path/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
