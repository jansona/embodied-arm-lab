"""Offline baseline evaluation; never calls Jev or reads credentials."""
import json,time
from pathlib import Path
from embodied_arm.simulation.simulator import Simulation
rows=[]
for seed in [0,1,2,3,42]:
    s=Simulation();s.reset(seed);t=time.perf_counter();s.baseline()
    for _ in range(2000):
        s.tick(4)
        if not s.current and not s.queue:break
    state=s.state()
    rows.append({'seed':seed,'controller':'scripted baseline (not Jev)','api_calls':0,'wall_seconds':round(time.perf_counter()-t,3),**state['metrics']})
    s.close()
result={'schema':'arm-evaluation-v1','jev_evaluated':False,'warning':'These are deterministic scripted-controller results, not model capability scores.','episodes':rows}
Path('artifacts').mkdir(exist_ok=True)
Path('artifacts/evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
