"""Single local-target API diagnostic using normal saved app configuration; no robot."""
import json
from pathlib import Path
from runner import JevRunner
class NoRobot:
    def stop(self):pass

def main():
    runner=JevRunner(NoRobot(),use_environment_network=True)
    runner.configure(Path(__file__).resolve().parent.parent/'jev-arm-config.json')
    state={'objective':{'id':'local_target','target_xyz':[.51,0,.3],'tolerance_m':.01},
           'end_effector':[.45,0,.3],'workspace':[[.25,.7],[-.35,.35],[.022,.6]]}
    result=runner._adapter.decide(state)
    print(json.dumps({'decision':result.to_dict(),'expected':'x_plus','correct':result.proposed_action=='x_plus','transport':runner._adapter.last_transport},indent=2))
if __name__=='__main__':main()
