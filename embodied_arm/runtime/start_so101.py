"""Start the SO101 preview or one explicitly bounded live session on loopback."""
from __future__ import annotations

import argparse
import os
import math
from pathlib import Path
import signal
import subprocess
import sys
import time

from embodied_arm.paths import WORKSPACE_ROOT, CONFIG_EXCLUSION_ROOT
ROOT = WORKSPACE_ROOT


def commands(*, live=False, config=None, workers_only=False, port=8766):
    """Build commands without loading credentials or calling a model."""
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('Port must be 1..65535')
    if workers_only and not live:
        raise ValueError('--workers-only requires --live')
    if live and not config:
        raise ValueError('--live requires an explicit --config outside the project')
    if config and not live:
        raise ValueError('--config requires --live')
    jobs = []
    if not workers_only:
        jobs.append(['-m', 'uvicorn', 'embodied_arm.web.app_so101:app', '--host', '127.0.0.1', '--port', str(port)])
    if live:
        path = Path(config).expanduser().resolve()
        if path == CONFIG_EXCLUSION_ROOT or CONFIG_EXCLUSION_ROOT in path.parents:
            raise ValueError('Configuration must remain outside the project')
        for script, calls, seconds in [('brain_worker.py', 6, 600),
                                        ('vision_worker.py', 6, 600),
                                        ('so101_worker.py', 120, 420)]:
            jobs.append(['-m', 'embodied_arm.runtime.' + script.removesuffix('.py'), '--config', str(path), '--max-calls', str(calls),
                         '--max-seconds', str(seconds)])
    return [[sys.executable, *job] for job in jobs]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Enable experimental simulator tasks and bounded workers; explicit UI requests may incur provider charges')
    parser.add_argument('--config', help='Existing provider configuration outside this repository')
    parser.add_argument('--workers-only', action='store_true', help='Only workers, when UI already runs in a supported execution context')
    parser.add_argument('--enable-waypoint', action='store_true', help='Enable experimental image-guided hover; not grasp or stack')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--camera-interval', type=float, default=1., help='Local RGB capture interval, 0.25..10 seconds; never periodic model calls')
    args = parser.parse_args(argv)
    if not math.isfinite(args.camera_interval) or not .25<=args.camera_interval<=10:
        parser.error('--camera-interval must be 0.25..10 seconds')
    try:
        jobs = commands(live=args.live, config=args.config, workers_only=args.workers_only, port=args.port)
    except ValueError as exc:
        parser.error(str(exc))
    children = []
    stop = False
    def interrupt(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    env = {**os.environ, 'SO101_ENABLE_WAYPOINT': '1' if args.enable_waypoint else '0', 'SO101_CAMERA_INTERVAL':str(args.camera_interval),
           'SO101_SIMULATION_TASKS':'1' if args.live else '0'}
    exit_code = 0
    try:
        for job in jobs:
            children.append(subprocess.Popen(job, cwd=ROOT, env=env))
        if not args.workers_only:
            print(f'SO101: http://127.0.0.1:{args.port} (loopback only)', flush=True)
        if args.live:
            print('Live window: Qwen 6 calls/600s; vision 6 calls/600s; Jev 120 calls/420s. No automatic renewal.', flush=True)
            print('Experimental simulator tasks: at most 120 Jev calls/300s per task; runtime guards may stop execution. No physical hardware control.', flush=True)
        else:
            print('Preview mode: simulator task execution disabled; no inference workers started.', flush=True)
        print('Ctrl+C stops all processes started by this launcher.', flush=True)
        reported_expiry = False
        while not stop:
            if any(child.poll() not in (None, 0) for child in children):
                print('A process failed; closing this session.', flush=True)
                exit_code = 1
                break
            if not args.workers_only and children[0].poll() is not None:
                break
            workers = children if args.workers_only else children[1:]
            if workers and all(child.poll() is not None for child in workers):
                if args.workers_only:
                    break
                if not reported_expiry:
                    print('Inference windows ended; preview remains open. Restart explicitly for another window.', flush=True)
                    reported_expiry = True
            time.sleep(.2)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
