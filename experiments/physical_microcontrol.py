"""Three-call numeric Jev closed-loop test in an independent physical simulator."""
import json,math,argparse
from pathlib import Path
from embodied_arm.legacy.panda.microcontrol_experiment import MicroController
from embodied_arm.legacy.panda.control import apply_action
from embodied_arm.simulation.simulator import Simulation
from embodied_arm.legacy.panda.runner import JevRunner
class NoRobot:
    def stop(self):pass

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--measured-band',action='store_true');args=parser.parse_args()
    runner=JevRunner(NoRobot(),use_environment_network=True)
    runner.configure(Path(__file__).resolve().parents[2]/'jev-arm-config.json')
    controller=MicroController(runner._adapter);sim=Simulation();rows=[]
    initial=sim.state()['ee'];target=[initial[0]+(.02 if args.measured_band else .04),initial[1],initial[2]]
    try:
        for index in range(2 if args.measured_band else 3):
            before=sim.state()['ee'];decision,metadata=controller.decide(before,target,'numeric',bool(index%2),distance_band=args.measured_band)
            row={'before':before,'target':target,'decision':decision.to_dict(),'transport':dict(runner._adapter.last_transport),**metadata,'executed':False}
            rows.append(row)
            if not decision.accepted:break
            if decision.action=='hold':
                row['at_target']=abs(target[0]-before[0])<=.01
                break
            apply_action(sim,decision.action,origin='jev microcontrol')
            while sim.state()['busy']:sim.tick(24)
            after=sim.state()['ee'];row.update(executed=True,after=after,distance_before=math.dist(before,target),distance_after=math.dist(after,target))
            print(json.dumps({'step':index+1,'action':decision.action,'confidence':decision.confidence,'distance_m':row['distance_after'],'elapsed_seconds':row['transport'].get('elapsed_seconds')}),flush=True)
            if row['distance_after']>=row['distance_before']:row['stopped_for_nonprogress']=True;break
        sim.stop()
        result={'calls':controller.calls,'initial':initial,'target':target,'final':sim.state()['ee'],'at_target':abs(target[0]-sim.state()['ee'][0])<=.01,'model_held':bool(rows and rows[-1]['decision']['accepted'] and rows[-1]['decision']['action']=='hold'),'steps':rows,'source':'real Jev selections; physical servo only; no deterministic action fallback'}
        Path(__file__).resolve().parents[1].joinpath('artifacts/physical-microcontrol-band.json' if args.measured_band else 'artifacts/physical-microcontrol.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='steps'}),flush=True)
    finally:controller.close();sim.close()
if __name__=='__main__':main()
