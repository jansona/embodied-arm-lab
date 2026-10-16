"""Independent known-fixture physics survey. NEVER a production controller.

Uses fixture ground truth solely to define deliberate offsets and score physical
contact, retention, and release. No models, network, credentials or fake grasps.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import pybullet as p
from embodied_arm.simulation.so101_simulation import SO101Simulation
from tests.test_so101_grasp_feasibility import _fixture_move, _settle


def trial(case):
    seed, object_name, offset = case[:3]
    close_steps = case[3] if len(case)>3 else 0
    fixture_yaw_deg = case[4] if len(case)>4 else 0
    trace_contacts = case[5] if len(case)>5 else False
    jaw_clearance_shift = case[6] if len(case)>6 else 0.
    planar_shift = case[7] if len(case)>7 else False
    sim=SO101Simulation(seed)
    result={'seed':seed,'object':object_name,'offset_mm':list(offset),
            'physics_fixture_only':True,'passed':False,'stage':'approach','close_ramp_ticks':close_steps,'fixture_yaw_deg':fixture_yaw_deg}
    try:
        body=sim._objects[object_name]
        if fixture_yaw_deg:
            # Explicit initial fixture configuration only, before any movement.
            initial=sim.call(p.getBasePositionAndOrientation,body)[0]
            sim.call(p.resetBasePositionAndOrientation,body,initial,
                     p.getQuaternionFromEuler([0,0,math.radians(fixture_yaw_deg)]))
            for _ in range(120):sim.call(p.stepSimulation)
        fixture=sim.evaluation_state()['ground_truth_objects'][object_name]
        jaws={sim.robot.link_indices['gripper_link'],sim.robot.gripper_joint_index}
        xyz=[fixture[0]+offset[0]/1000,fixture[1]+offset[1]/1000,.012+offset[2]/1000]
        if jaw_clearance_shift:
            # Deterministic geometry recipe: move TCP toward its fixed jaw,
            # placing the target farther into the open aperture. Direction
            # derives from robot FK only, never object/contact outcomes.
            pose=sim.robot.forward_kinematics(sim.robot.solve_ik(xyz))
            axis=pose[:3,0].copy()
            if planar_shift:
                axis[2]=0.;axis/=math.hypot(axis[0],axis[1])
                result['planar_shift']=True
            xyz=[v+jaw_clearance_shift*a for v,a in zip(xyz,axis)]
            result['jaw_clearance_shift_m']=jaw_clearance_shift
            result['jaw_axis_world']=pose[:3,0].tolist()
        # Motion helper only sends bounded facade servo commands. No reset,
        # constraint, collision disable, object relocation or motor-force change.
        if trace_contacts:
            result['descent_contacts']=[]
            original_tick=sim.tick
            recorded=set()
            def traced_tick(count=4):
                for _ in range(count):
                    original_tick(1)
                    if result['stage']!='descend':continue
                    for c in sim.call(p.getContactPoints,sim.robot.body_id,body):
                        if c[9]<=0 or c[3] in recorded:continue
                        recorded.add(c[3])
                        names={v:k for k,v in sim.robot.link_indices.items()}
                        result['descent_contacts'].append({'link':names.get(c[3],str(c[3])),
                            'normal_force_n':c[9],'contact_robot_xyz':c[5],
                            'tcp_xyz':sim.robot.tcp_pose()[0]})
            sim.tick=traced_tick
        sim.resume()
        _fixture_move(sim,[xyz[0],xyz[1],.06])
        result['stage']='descend'
        _fixture_move(sim,xyz)
        result['stage']='close'
        before_close=sim.evaluation_state()['ground_truth_objects'][object_name]
        result['preclose_shift_mm']=math.dist(before_close[:2],fixture[:2])*1000
        result['preclose_xyz']=before_close
        if close_steps:
            start=sim.proprioception()['gripper_openness']
            for index in range(close_steps):
                t=(index+1)/close_steps
                sim.robot.command_gripper(start*(1-t*t*(3-2*t)))
                sim.tick(1)
            _settle(sim,360)
        else:
            # Preserve the original step-input baseline even after the facade
            # gained production gripper interpolation. No object manipulation.
            sim.robot.command_gripper(0);_settle(sim,360)
        after_close=sim.evaluation_state()['ground_truth_objects'][object_name]
        result['postclose_shift_mm']=math.dist(after_close[:2],fixture[:2])*1000
        result['postclose_xyz']=after_close
        contacts=sim.call(p.getContactPoints,sim.robot.body_id,body)
        result['two_jaw_close']=jaws <= {c[3] for c in contacts if c[9]>0}
        result['stage']='lift'
        _fixture_move(sim,[xyz[0],xyz[1],.06])
        _settle(sim,480)
        lifted=sim.evaluation_state()['ground_truth_objects'][object_name]
        contacts=sim.call(p.getContactPoints,sim.robot.body_id,body)
        result['two_jaw_hold']=jaws <= {c[3] for c in contacts if c[9]>0}
        result['lift_mm']=(lifted[2]-fixture[2])*1000
        result['lateral_shift_mm']=math.dist(lifted[:2],fixture[:2])*1000
        result['constraints']=sim.call(p.getNumConstraints)
        result['stage']='release'
        sim.manual('open',origin='offline_fixture');_settle(sim,720)
        released=sim.evaluation_state()['ground_truth_objects'][object_name]
        result['release_z_mm']=released[2]*1000
        result['passed']=(result['two_jaw_close'] and result['two_jaw_hold']
                          and result['lift_mm']>35 and result['lateral_shift_mm']<20
                          and result['constraints']==0 and released[2]<.015)
        result['stage']='complete'
    except Exception as exc:
        result['error']=str(exc)
    finally:sim.close()
    return result


def cases(profile):
    shift=math.sqrt(2)*.01+.006+.002-.012
    if profile=='slowclose':
        return [(19,'red',offset,ramp) for offset in [(2,0,0),(2,-2,0),(0,4,0),(0,0,0)] for ramp in (0,144,360,720)]
    if profile=='yaw':
        return [(19,name,offset,0,yaw) for name in ('red','blue') for offset in [(0,0,0),(2,0,0)] for yaw in (0,15,30,45,60,75)]
    if profile=='shifted_yaw':
        return [(19,name,offset,360,yaw,False,shift) for name in ('red','blue') for offset in [(0,0,0),(6,6,0)] for yaw in (0,15,30,45,60,75)]
    if profile in ('shifted','shifted_z4','shifted_zlowplanar'):
        zs={'shifted':(-2,0,2),'shifted_z4':(-4,4),'shifted_zlowplanar':(-4,)}[profile]
        return [(seed,name,(x,y,z),360,0,False,shift,profile=='shifted_zlowplanar')
                for seed in (19,47) for name in ('red','blue') for x in (-6,0,6) for y in (-6,0,6) for z in zs]
    if profile=='centered3':
        return [(seed,name,(x,y,z),360,0,False,shift,True) for seed in (19,47)
                for name in ('red','blue') for x in (-3,0,3) for y in (-3,0,3) for z in (-2,1,4)]
    if profile=='contacttrace':
        return [(19,'red',offset,0,0,True) for offset in [(0,0,0),(2,0,0)]]
    if profile=='axes':
        offsets=[(0,0,0)]
        for axis,values in ((0,(-8,-6,-4,-2,2,4,6,8)),
                            (1,(-8,-6,-4,-2,2,4,6,8)),(2,(-4,-2,2,4))):
            for value in values:
                offset=[0,0,0];offset[axis]=value;offsets.append(tuple(offset))
    else:
        offsets=[(sx*r,sy*r,z) for r in (2,4,6) for sx in (-1,1)
                 for sy in (-1,1) for z in (-2,0,2)]
    return [(seed,name,offset) for seed in (19,47) for name in ('red','blue') for offset in offsets]


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['axes','corners','slowclose','yaw','shifted','shifted_yaw','shifted_z4','shifted_zlowplanar','contacttrace','centered3'],default='axes')
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--output',required=True)
    args=parser.parse_args();results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(trial,cases(args.profile)):
            results.append(result)
            print(json.dumps(result),flush=True)
    Path(args.output).write_text(json.dumps({'scope':'independent_known_fixture_physics_only',
            'profile':args.profile,'cases':results},indent=2)+'\n')
