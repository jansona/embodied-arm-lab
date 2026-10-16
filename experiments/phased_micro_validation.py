"""At most eight real Jev requests; isolated physical XYZ move/hold trials."""
import json,math
from pathlib import Path
from simulator import Simulation
from control import apply_action
from runner import JevRunner
from microcontrol_experiment import MicroController
class NoRobot:
    def stop(self):pass

def main():
    runner=JevRunner(NoRobot(),use_environment_network=True)
    runner.configure(Path(__file__).resolve().parents[2]/'jev-arm-config.json')
    controller=MicroController(runner._adapter);trials=[]
    try:
        for trial_index,(axis,sign) in enumerate([('x',1),('x',-1),('y',1),('z',1)]):
            sim=Simulation();initial=sim.state()['ee'];target=list(initial);target['xyz'.index(axis)]+=.02*sign;steps=[]
            try:
                for step_index in range(2):
                    before=sim.state()['ee'];inside=max(abs(a-b) for a,b in zip(before,target))<=.01
                    decision,metadata=controller.decide(before,target,'numeric',bool((trial_index+step_index)%2),True,axis=axis)
                    row={'before':before,'target':target,'decision':decision.to_dict(),'transport':dict(runner._adapter.last_transport),**metadata,'executed':False,'terminal_guard_stop':False}
                    steps.append(row)
                    if not decision.accepted:break
                    if decision.action=='hold':row['correct_hold']=inside;break
                    if inside:row['terminal_guard_stop']=True;break
                    apply_action(sim,decision.action,origin='jev phased microcontrol')
                    while sim.state()['busy']:sim.tick(24)
                    row.update(executed=True,after=sim.state()['ee'],distance_before=math.dist(before,target),distance_after=math.dist(sim.state()['ee'],target))
                    if row['distance_after']>=row['distance_before']:row['nonprogress_stop']=True;break
                sim.stop();result={'axis':axis,'sign':sign,'initial':initial,'target':target,'final':sim.state()['ee'],'at_target':max(abs(a-b) for a,b in zip(sim.state()['ee'],target))<=.01,'model_held':bool(steps and steps[-1].get('correct_hold')),'steps':steps}
                trials.append(result)
                print(json.dumps({'trial':trial_index+1,'axis':axis,'sign':sign,'at_target':result['at_target'],'model_held':result['model_held'],'choices':[s['decision']['proposed_action'] for s in steps],'confidence':[s['decision']['confidence'] for s in steps],'latencies':[s['transport'].get('elapsed_seconds') for s in steps]}),flush=True)
                if not(result['at_target'] and result['model_held']):break
            finally:sim.close()
        output={'calls':controller.calls,'trials':trials,'source':'real Jev chooses sign and hold; fixed axis phase specified by test; terminal guard never replaces an action'}
        Path(__file__).resolve().parents[1].joinpath('artifacts/phased-micro-validation.json').write_text(json.dumps(output,indent=2))
        print(json.dumps({'calls':controller.calls,'passed':sum(t['at_target'] and t['model_held'] for t in trials),'trials':len(trials)}),flush=True)
    finally:controller.close()
if __name__=='__main__':main()
