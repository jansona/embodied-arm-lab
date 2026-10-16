"""One fresh, provenance-checked original s4 continuation; no new Qwen call."""
from embodied_arm.paths import RUNTIME_ROOT
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time
from embodied_arm.providers.brain import validate_plan
from embodied_arm.legacy.panda.continue_stack import ROOT, HTTPArmBackend, GuardedController, _read
from embodied_arm.runtime.ipc import FileBridgeAdapter
from embodied_arm.legacy.panda.stack_plans import plan_for_symbolic_step, symbolic_step_result
from embodied_arm.legacy.panda.task_executor import TaskExecutor

MAX_CALLS=70
MAX_SECONDS=180

def load_final_checkpoint(snapshot_path,provenance_path,prior_audit_path):
    raw,state=_read(snapshot_path);_,p=_read(provenance_path);audit_raw,audit=_read(prior_audit_path)
    expected={'plan','source_call_id','model','provenance','accepted','snapshot_sha256',
              'completed_step_ids','continuation_audit_sha256'}
    if set(p)!=expected or p['snapshot_sha256']!=hashlib.sha256(raw).hexdigest() or p['continuation_audit_sha256']!=hashlib.sha256(audit_raw).hexdigest():
        raise ValueError('Fresh checkpoint provenance mismatch')
    if p['model']!='qwen3.8-max' or p['provenance']!='live' or p['accepted'] is not True or p['completed_step_ids']!=['s1','s2','s3']:
        raise ValueError('Invalid completed prefix')
    plan=validate_plan(p['plan'],state,check_state=False).to_dict()
    if plan!=audit['original_plan'] or p['source_call_id']!=audit['source_call_id'] or audit['completed_prefix']!=['s1']:
        raise ValueError('Original accepted plan does not match')
    if len(plan['steps'])!=4 or plan['steps'][-1]!={'step_id':'s4','action':'place','object_id':'blue','target_id':'on:red'}:
        raise ValueError('Only original final blue-on-red step is supported')
    if not any(x['source_step_id']=='s2' and x['outcome'].get('verified') is True for x in audit['remaining_step_results']):
        raise ValueError('Red placement was not verified')
    completed=audit['active_task']['steps'][0]
    if completed['action']!='pick' or completed['object_id']!='blue' or completed['status']!='completed' or not completed['outcome'].get('verified'):
        raise ValueError('Blue lift was not verified')
    contacts=state.get('finger_contacts_by_object',{}).get('blue',{})
    if (not state['paused'] or state['busy'] or state.get('held_object')!='blue'
        or not all(contacts.get(str(i),0)>0 for i in (9,10))
        or math.dist(state['blocks']['red'],state['goal'])>.03):
        raise ValueError('Fresh physical state is not safe for final placement')
    plan_for_symbolic_step(plan['steps'][-1],state)
    return state,p

def run_final(state,p,backend,controller,deadline,audit_path,executor_factory=TaskExecutor):
    step=p['plan']['steps'][-1]
    report={'schema':'stack-final-continuation-v1','provenance':'reused_live','source_call_id':p['source_call_id'],
        'snapshot_sha256':p['snapshot_sha256'],'continuation_audit_sha256':p['continuation_audit_sha256'],
        'original_step':step,'completed_prefix':['s1','s2','s3'],'max_calls':MAX_CALLS,'max_seconds':MAX_SECONDS,
        'new_qwen_calls':0,'controller':'jev','status':'starting','verified':False,
        'ui_note':'Original view retains interrupted task; actuator manual origin is legacy metadata.'}
    executor=None
    def save():
        report.update(jev_calls=controller.calls,decisions=controller.decisions)
        path=Path(audit_path);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)
    try:
        primitive=plan_for_symbolic_step(step,backend.state())
        executor=executor_factory(backend,controller=controller)
        executor.start(primitive,max_seconds=120,max_actions=100);report['status']='running';last=-1
        while executor.status()['status']=='running':
            if backend.cancelled or time.monotonic()>=deadline:raise RuntimeError('Bounded continuation stopped')
            backend.state()
            if last!=controller.calls:last=controller.calls;report['task']=executor.status();save()
            time.sleep(.05)
        report['task']=executor.status()
        if report['task']['status']!='completed':raise RuntimeError('Controller or task stopped')
        after=backend.state();outcome=symbolic_step_result(step,state,after)
        report.update(outcome=outcome,verified=bool(outcome.get('verified')),final_scene=after,
                      status='completed' if outcome.get('verified') else 'stopped')
    except Exception:
        if executor:report['task']=executor.status();executor.stop()
        report.update(status='stopped',reason=getattr(backend,'failure_reason',None) or 'task_or_controller_stop')
    finally:controller.cancel();backend.stop();save()
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('snapshot','plan','prior-audit','audit'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();lock=(RUNTIME_ROOT/'stack-continuation.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state,p=load_final_checkpoint(args.snapshot,args.plan,args.prior_audit)
    backend=HTTPArmBackend(state);adapter=FileBridgeAdapter()
    if not adapter.status().get('available'):raise RuntimeError('Jev worker unavailable; no motion started')
    deadline=time.monotonic()+MAX_SECONDS;controller=GuardedController(adapter,backend,deadline,max_calls=MAX_CALLS)
    try:
        result=run_final(state,p,backend,controller,deadline,args.audit)
        print(json.dumps({k:result.get(k) for k in ('status','verified','jev_calls','new_qwen_calls','reason')}))
    finally:backend.close()
if __name__=='__main__':main()
