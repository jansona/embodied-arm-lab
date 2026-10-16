"""Independent known-physics fixture with image-only perception checks.

Fixture targets use known scene positions; RGBColourFixture/CameraSceneSource
receive ONLY JPEG pixels, calibrations and encoders. No model calls or GT inputs.
"""
import json,math
from dataclasses import replace
from pathlib import Path
from embodied_arm.simulation.so101_simulation import SO101Simulation
from embodied_arm.control.so101_control import CameraSceneSource
from tests.rgb_coarse_fixture import RGBColourFixture
from tests.test_so101_grasp_feasibility import _fixture_move,_settle


class ScopedRGBFixture(RGBColourFixture):
    def perceive(self,packet):
        result=super().perceive(packet)
        return replace(result,detections=tuple(d for d in result.detections if d.object_id in ('red','blue')))


def run(name, *, scoped=False):
    sim=SO101Simulation(0,camera_size=(1920,1440),illumination_profile='shadow_free_rgb_v1')
    rows=[]
    try:
        # Independent trajectory fixture only, never passed to either perception component.
        fixture=sim.evaluation_state()['ground_truth_objects'][name]
        xyz=[fixture[0],fixture[1],.012]
        pose=sim.robot.forward_kinematics(sim.robot.solve_ik(xyz))
        shift=math.sqrt(2)*.01+.006+.002-.012
        xyz=[v+shift*a for v,a in zip(xyz,pose[:3,0])]
        source=CameraSceneSource(sim,(ScopedRGBFixture if scoped else RGBColourFixture)(profile='scaled'),refine_rgb=True,
                                 refinement_profile='scaled')
        def observe(stage):
            sim.pause_idle();row={'object':name,'stage':stage,'perception_source':'offline_fixture_rgb_geometry'}
            try:
                snapshot=source.refresh(required_objects=('red','blue'))
                row.update(accepted=True,scene=snapshot['scene'])
            except Exception as exc:
                status=source.status()
                row.update(accepted=False,error=str(exc),reason=status.get('reason'),refinement=status.get('last_vision'))
            for camera in ('left','right'):
                path=Path('/tmp')/f'so101-shifted-{name}-{stage}-{camera}.jpg'
                path.write_bytes(sim.render(camera));row[camera+'_image']=str(path)
            rows.append(row)
            print(json.dumps({'object':name,'stage':stage,'accepted':row['accepted'],'error':row.get('error')}),flush=True)
            sim.resume()
        observe('initial')
        _fixture_move(sim,[xyz[0],xyz[1],.06]);observe('approach')
        _fixture_move(sim,xyz);observe('preclose')
        for index in range(360):
            t=(index+1)/360;sim.robot.command_gripper(1-t*t*(3-2*t));sim.tick(1)
        _settle(sim,360);observe('closed')
        _fixture_move(sim,[xyz[0],xyz[1],.06]);_settle(sim,480);observe('lifted')
        return rows
    finally:sim.close()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('artifacts/so101-grasp-visibility.json'))
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in ('red','blue'):
        rows.extend(run(name))
        args.output.write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps([{'object':r['object'],'stage':r['stage'],'accepted':r['accepted'],'reason':r.get('reason')} for r in rows]),flush=True)
