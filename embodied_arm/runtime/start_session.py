"""Explicit bounded session launcher. No automatic restart or credential output."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from embodied_arm.paths import WORKSPACE_ROOT
ROOT=WORKSPACE_ROOT

def commands(workers_only=False):
    jobs=[['-m','embodied_arm.runtime.brain_worker','--config',str(ROOT.parent/'jev-arm-config.json'),'--max-calls','6','--max-seconds','600'],
          ['-m','embodied_arm.runtime.inference_worker','--micro-phased','--max-calls','200','--max-seconds','420']]
    if not workers_only:jobs.insert(0,['-m','uvicorn','embodied_arm.web.app:app','--host','127.0.0.1','--port','8765'])
    return [[sys.executable,*args] for args in jobs]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers-only',action='store_true',help='For the approved network environment when the desktop already runs ./run-ipc.sh')
    args=parser.parse_args();children=[];stop=False;reported_expiry=False
    def interrupt(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
    env={**os.environ,'JEV_USE_IPC':'1'}
    try:
        for command in commands(args.workers_only):children.append(subprocess.Popen(command,cwd=ROOT,env=env))
        print('Explicit session: brain <=6 calls/600s; Jev <=200 calls/420s. No automatic renewal. Ctrl+C stops these processes.',flush=True)
        while not stop:
            if any(p.poll() not in (None,0) for p in children):break
            workers=children if args.workers_only else children[1:]
            if all(p.poll() is not None for p in workers):
                if args.workers_only:break
                if not reported_expiry:
                    print('Inference window ended; simulator remains available. No automatic renewal.',flush=True);reported_expiry=True
            if not args.workers_only and children[0].poll() is not None:break
            time.sleep(.2)
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait()
if __name__=='__main__':main()
