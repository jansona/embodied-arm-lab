"""Explicit, bounded inference worker for fixed simulation-only file IPC."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from embodied_arm.paths import WORKSPACE_ROOT
import queue
import threading
import time
from embodied_arm.providers.adapters import Decision, sanitize_transport
from embodied_arm.runtime.ipc import Store, number, validate_state, validate_decision


class AuditRecorder:
    """Bounded diagnostic data; logging failure must never interrupt control."""
    def __init__(self,store):self.store=store;self.recent=[];self.summary=[];self.errors=0
    def record(self,entry):
        self.recent.append(entry);self.recent=self.recent[-8:]
        decision=entry['decision']
        self.summary.append({k:decision.get(k) for k in ('action','proposed_action','confidence','accepted','reason')} | {k:entry.get(k) for k in ('latency_seconds','delivered')})
        self.summary=self.summary[-128:]
        try:
            self.store.write('last-run-audit.json',self.recent)
            self.store.write('run-summary.json',self.summary)
        except (OSError,ValueError,TypeError):self.errors+=1


def worker_run(adapter,max_calls=8,max_seconds=30):
    if type(max_calls) is not int or not 1<=max_calls<=getattr(adapter,'max_worker_calls',8): raise ValueError('Call budget exceeds this controller limit')
    if not number(max_seconds) or not 0<max_seconds<=getattr(adapter,'max_worker_seconds',30): raise ValueError('Time budget exceeds this controller limit')
    store=Store();lock_fd=None;started=time.monotonic();deadline=started+max_seconds
    counters={'status':'ready','calls':0,'processed':0,'rejected':0,'cancelled':0,'elapsed_seconds':0.0,'max_calls':max_calls,'max_seconds':max_seconds,'deadline_unix':time.time()+max_seconds}
    locked=False; audit=AuditRecorder(store)
    def publish(status):
        counters['status']=status;counters['elapsed_seconds']=round(time.monotonic()-started,3)
        store.write('ready.json',counters)
    try:
        lock_fd=os.open('worker.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600,dir_fd=store.fd)
        try: fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('An IPC worker is already active') from None
        locked=True;store.write('last-run-audit.json',[]);store.write('run-summary.json',[]);publish('ready')
        while counters['calls']<max_calls and time.monotonic()<deadline:
            jobs=store.jobs()
            if not jobs: time.sleep(.02);continue
            for name in jobs:
                if counters['calls']>=max_calls or time.monotonic()>=deadline: break
                job=name[:-4]
                try:
                    data=store.read(name)
                    if not isinstance(data,dict) or set(data)!={'id','expires_at','state'} or data['id']!=job or not number(data['expires_at']): raise ValueError('Invalid job')
                    state=validate_state(data['state'])
                    if state['busy'] or state['paused']: raise ValueError('Only idle simulation observations allowed')
                    if not 0<data['expires_at']-time.time()<=20.5:
                        counters['rejected']+=1;store.delete(name);continue
                    if store.exists(job+'.cancel'):
                        counters['cancelled']+=1;store.delete(name);continue
                except (ValueError,TypeError,OSError,RecursionError):
                    counters['rejected']+=1;store.delete(name);continue
                # No worker-supplied destinations, headers, or credentials exist
                # in this protocol. Adapter/configuration owns the fixed provider.
                result_queue=queue.Queue(maxsize=1)
                def invoke(snapshot=state,results=result_queue):
                    try: result=adapter.decide(snapshot)
                    except Exception: result=Decision(reason='adapter_error')
                    results.put(result)
                if store.exists(job+'.cancel'):
                    counters['cancelled']+=1;store.delete(name);continue
                counters['calls']+=1;publish('calling');call_started=time.monotonic()
                threading.Thread(target=invoke,daemon=True,name='bounded-ipc-inference').start()
                result=None
                while time.monotonic()<deadline:
                    try: result=result_queue.get(timeout=min(.02,max(.001,deadline-time.monotonic())));break
                    except queue.Empty: pass
                if result is None:
                    audit.record({'state':state,'decision':Decision(reason='timeout').to_dict(),'latency_seconds':round(time.monotonic()-call_started,4),'delivered':False,'transport':sanitize_transport(getattr(adapter,'last_transport',{}),max_events=12)})
                    store.write(job+'.cancel',{'cancelled':True})
                    store.delete(name)
                    break
                counters['processed']+=1
                try:
                    if not isinstance(result,Decision): raise ValueError('Invalid adapter result')
                    safe=result.to_dict()
                    if not result.accepted and result.reason not in {'low_confidence','timeout','network_error','invalid_response','invalid_state','response_too_large','adapter_error'}:
                        safe=Decision(reason='api_error').to_dict()
                    validate_decision(safe)
                except (ValueError,TypeError): safe=Decision(reason='invalid_response').to_dict()
                cancelled=store.exists(job+'.cancel') or time.time()>=data['expires_at']
                audit.record({'state':state,'decision':safe,'latency_seconds':round(time.monotonic()-call_started,4),'delivered':not cancelled,'transport':sanitize_transport(getattr(adapter,'last_transport',{}),max_events=12)})
                if cancelled:
                    counters['cancelled']+=1
                else:
                    store.write(job+'.result',safe)
                store.delete(name);publish('ready')
        publish('call_budget' if counters['calls']>=max_calls else 'time_budget')
        counters['audit_write_errors']=audit.errors
        return dict(counters)
    finally:
        if lock_fd is not None:
            if locked: fcntl.flock(lock_fd,fcntl.LOCK_UN)
            os.close(lock_fd)
        store.close()


class NoRobot:
    def stop(self): pass


def main():
    parser=argparse.ArgumentParser(description='Run one bounded simulation-only inference IPC window')
    parser.add_argument('--max-calls',type=int,default=8)
    parser.add_argument('--max-seconds',type=float,default=30)
    parser.add_argument('--micro-x',action='store_true',help='Experimental pooled local-X controller')
    parser.add_argument('--micro-phased',action='store_true',help='Pooled fixed XYZ phase decomposition with Jev movement and gripper choices')
    args=parser.parse_args()
    # Configuration is loaded by the existing approved loader, never printed or
    # copied into IPC. No environment-secret discovery or generic transport.
    from embodied_arm.legacy.panda.runner import JevRunner
    runner=JevRunner(NoRobot(),use_environment_network=True)
    try:
        runner.configure(WORKSPACE_ROOT.parent/'jev-arm-config.json')
        if runner._adapter is None: raise ValueError('Not configured')
        adapter=runner._adapter
        if args.micro_x and args.micro_phased:raise ValueError('Choose one micro mode')
        if args.micro_phased:
            from embodied_arm.legacy.panda.microcontrol_experiment import PhasedMicroAdapter
            adapter=PhasedMicroAdapter(adapter)
        if args.micro_x:
            from embodied_arm.legacy.panda.microcontrol_experiment import XMicroAdapter
            adapter=XMicroAdapter(adapter)
        try: print(json.dumps(worker_run(adapter,args.max_calls,args.max_seconds)))
        finally:
            if args.micro_x or args.micro_phased:adapter.close()
    except Exception:
        print(json.dumps({'status':'worker_error'}))
        raise SystemExit(1) from None


if __name__=='__main__': main()
