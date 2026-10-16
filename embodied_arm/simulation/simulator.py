"""CPU-only Panda physics sandbox. No grasp constraints or object teleporting."""
import math
import threading
import time
from collections import deque
from io import BytesIO

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

HOME = [0, -0.4, 0, -2.2, 0, 1.8, 0.785398]
WORKSPACE = ((0.25, 0.7), (-0.35, 0.35), (0.022, 0.6))

class Simulation:
    def __init__(self):
        self.lock = threading.RLock()
        self.cid = p.connect(p.DIRECT)
        self.reset()

    def call(self, fn, *args, **kwargs):
        return fn(*args, physicsClientId=self.cid, **kwargs)

    def reset(self, seed=0):
        with self.lock:
            self._render_cache=None;self._render_at=0.0
            self.task_mode="hover_reach"
            self.seed=int(seed)
            rng=np.random.default_rng(self.seed)
            red_position=[.5,0,.025] if self.seed==0 else [float(rng.uniform(.47,.54)),float(rng.uniform(-.045,.045)),.025]
            self.call(p.resetSimulation)
            self.call(p.setAdditionalSearchPath, pybullet_data.getDataPath())
            self.call(p.setGravity, 0, 0, -9.81)
            self.call(p.setTimeStep, 1/240)
            self.call(p.setPhysicsEngineParameter, numSolverIterations=150, deterministicOverlappingPairs=1)
            self.floor = self.call(p.loadURDF, 'plane.urdf')
            self.robot = self.call(p.loadURDF, 'franka_panda/panda.urdf', useFixedBase=True)
            self.limits = [(self.call(p.getJointInfo,self.robot,j)[8],self.call(p.getJointInfo,self.robot,j)[9]) for j in range(7)]
            for j,q in enumerate(HOME): self.call(p.resetJointState,self.robot,j,q)
            for j in (9,10):
                self.call(p.resetJointState,self.robot,j,0.04)
                self.call(p.changeDynamics,self.robot,j,lateralFriction=3,spinningFriction=0.1)
            # The Panda URDF specifies finger_joint2 mimics finger_joint1. Bullet
            # needs this robot-only linkage explicitly; independent saturated
            # jaw motors otherwise translate together and carry a held cube
            # sideways despite unchanged total aperture. Never attach an object.
            # Same linkage/settings as Bullet's official panda_sim_grasp example.
            self.finger_constraint=self.call(p.createConstraint,self.robot,9,self.robot,10,
                                              p.JOINT_GEAR,[1,0,0],[0,0,0],[0,0,0])
            self.call(p.changeConstraint,self.finger_constraint,gearRatio=-1,erp=.1,maxForce=50)
            self.blocks = {}
            for name,pos,color in [('red',red_position,[.94,.22,.24,1]),('blue',[.58,-.19,.025],[.2,.48,.98,1])]:
                cs=self.call(p.createCollisionShape,p.GEOM_BOX,halfExtents=[.025]*3)
                vs=self.call(p.createVisualShape,p.GEOM_BOX,halfExtents=[.025]*3,rgbaColor=color)
                body=self.call(p.createMultiBody,baseMass=.06,baseCollisionShapeIndex=cs,baseVisualShapeIndex=vs,basePosition=pos)
                self.call(p.changeDynamics,body,-1,lateralFriction=2,spinningFriction=.01,rollingFriction=.001)
                self.blocks[name]=body
            marker=self.call(p.createVisualShape,p.GEOM_BOX,halfExtents=[.065,.065,.001],rgbaColor=[.18,.75,.52,0.8])
            self.call(p.createMultiBody,baseMass=0,baseVisualShapeIndex=marker,basePosition=[.48,.21,.001])
            self.goal=[.48,.21,.025]
            self.queue=deque(); self.current=None; self.paused=False; self.mode='manual'; self.status='Ready'; self.error=None
            self.steps=0;self.actions=0;self.max_height=.025;self.origin='reset';self.completed=False
            self.joint_target=list(HOME);self.grip=.04
            self.motor()
            for _ in range(120):self.call(p.stepSimulation)
            self.target=list(self.ee())

    def ee(self): return self.call(p.getLinkState,self.robot,11,computeForwardKinematics=True)[4]

    def motor(self):
        self.call(p.setJointMotorControlArray,self.robot,list(range(7)),p.POSITION_CONTROL,targetPositions=self.joint_target,forces=[87,87,87,87,12,12,12],positionGains=[.12]*7,velocityGains=[1]*7)
        self.call(p.setJointMotorControlArray,self.robot,[9,10],p.POSITION_CONTROL,targetPositions=[self.grip]*2,forces=[40,40],positionGains=[.15]*2)

    def validate_xyz(self, xyz):
        if len(xyz)!=3 or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in xyz):raise ValueError('XYZ must contain three finite numbers')
        if not all(lo<=v<=hi for v,(lo,hi) in zip(xyz,WORKSPACE)):raise ValueError('Target outside workspace')

    def ik(self, xyz):
        self.validate_xyz(xyz)
        qs=self.call(p.calculateInverseKinematics,self.robot,11,xyz,p.getQuaternionFromEuler([math.pi,0,0]),maxNumIterations=120,residualThreshold=1e-5)
        if not all(lo-1e-4 <= q <= hi+1e-4 for q,(lo,hi) in zip(qs[:7],self.limits)): raise ValueError('IK exceeds joint limits')
        return list(qs[:7])

    def enqueue(self, xyz=None, grip=None, duration=1.0, label='Move', origin='manual'):
        if xyz is not None:self.validate_xyz(xyz)
        self.queue.append({'xyz':xyz,'grip':grip,'ticks':max(1,int(duration*240)), 'label':label,'origin':origin})

    def manual(self, action, values=None):
        with self.lock:
            if self.paused:raise ValueError('Resume before moving')
            if self.queue or self.current:raise ValueError('Motion busy; stop first')
            self.mode='manual'; self.completed=False
            if action=='xyz':
                self.validate_xyz(values)
                if math.dist(self.ee(),values)>.101:raise ValueError('Manual move limited to 10 cm')
                self.ik(values)
                self.enqueue(list(values),duration=.8)
            elif action=='joints':
                if len(values)!=7 or not all(math.isfinite(v) and lo<=v<=hi for v,(lo,hi) in zip(values,self.limits)):raise ValueError('Invalid joint target')
                now=[self.call(p.getJointState,self.robot,j)[0] for j in range(7)]
                if max(abs(a-b) for a,b in zip(now,values))>.251:raise ValueError('Joint step limited to 0.25 radians')
                self.queue.append({'joints':values,'ticks':192,'label':'Joint control','origin':'manual'})
            elif action in ('open','close'):self.enqueue(grip=.04 if action=='open' else 0,duration=.7,label=action)
            else:raise ValueError('Unknown action')
            self.actions+=1

    def baseline(self):
        with self.lock:
            if self.paused or self.current or self.queue:raise ValueError('Stop or reset before starting')
            self.mode='scripted baseline (not Jev)';self.completed=False
            b=self.call(p.getBasePositionAndOrientation,self.blocks['red'])[0]
            for xyz,grip,duration,label in [([b[0],b[1],.25],.04,1.5,'Approach red block'),([b[0],b[1],.029],.04,1.8,'Lower gripper'),(None,0,1,'Close fingers'),([b[0],b[1],.27],0,1.8,'Lift with physical contact'),([self.goal[0],self.goal[1],.27],0,1.8,'Carry to target'),([self.goal[0],self.goal[1],.031],0,1.8,'Lower to target'),(None,.04,1,'Release'),([self.goal[0],self.goal[1],.28],.04,1.5,'Retract')]:
                self.enqueue(xyz,grip,duration,label,'scripted baseline')
            self.actions+=8

    def stop(self):
        with self.lock:
            self.queue.clear();self.current=None;self.paused=True;self.status='Stopped';self.mode='manual'
            self.joint_target=[self.call(p.getJointState,self.robot,j)[0] for j in range(7)];self.motor()

    def tick(self, count=4):
        with self.lock:
            if self.paused:return
            for _ in range(count):
                if self.current is None and self.queue:
                    cmd=self.queue.popleft();cmd['elapsed']=0;cmd['start']=list(self.ee());cmd['start_joints']=list(self.joint_target)
                    self.current=cmd;self.status=cmd['label'];self.origin=cmd['origin']
                    if cmd.get('grip') is not None:self.grip=cmd['grip']
                if self.current:
                    c=self.current;c['elapsed']+=1;t=min(1,c['elapsed']/c['ticks']); t=t*t*(3-2*t)
                    try:
                        if c.get('xyz') is not None:
                            self.target=[a+(b-a)*t for a,b in zip(c['start'],c['xyz'])]
                            self.joint_target=self.ik(self.target)
                        elif 'joints' in c:self.joint_target=[a+(b-a)*t for a,b in zip(c['start_joints'],c['joints'])]
                        self.motor()
                    except ValueError as e:
                        self.error=str(e);self.stop();return
                    if c['elapsed']>=c['ticks']:
                        self.current=None
                        if not self.queue:self.status='Motion complete';self.completed=True
                self.call(p.stepSimulation);self.steps+=1
                self.max_height=max(self.max_height,self.call(p.getBasePositionAndOrientation,self.blocks['red'])[0][2])

    def state(self):
        with self.lock:
            blocks={name:list(self.call(p.getBasePositionAndOrientation,body)[0]) for name,body in self.blocks.items()}
            red=blocks['red']; ee=list(self.ee())
            contacts_by_object={name:{str(j):len(self.call(p.getContactPoints,self.robot,body,linkIndexA=j)) for j in (9,10)} for name,body in self.blocks.items()}
            contacts=contacts_by_object['red']
            width=sum(self.call(p.getJointState,self.robot,j)[0] for j in (9,10))
            held=[name for name,xyz in blocks.items() if xyz[2]>.13 and math.dist(xyz,ee)<.065 and 0<=width<=.065]
            held_object=held[0] if len(held)==1 else None
            return {'held_object':held_object,'finger_contacts_by_object':contacts_by_object,'task_mode':self.task_mode,'task_description':'Limited waypoint test: move toward high hover above red block; no grasp','seed':self.seed,'engine':'PyBullet DIRECT / TinyRenderer','mode':self.mode,'origin':self.origin,'status':self.status,'error':self.error,'paused':self.paused,'busy':bool(self.current or self.queue),'ee':ee,'target':self.target,'joints':[self.call(p.getJointState,self.robot,j)[0] for j in range(7)],'joint_limits':self.limits,'gripper_width':sum(self.call(p.getJointState,self.robot,j)[0] for j in (9,10)),'blocks':blocks,'goal':self.goal,'workspace':WORKSPACE,'metrics':{'physics_steps':self.steps,'simulation_seconds':round(self.steps/240,2),'actions':self.actions,'red_max_height_m':self.max_height,'goal_distance_m':math.dist(red,self.goal),'reach_distance_m':math.dist(ee,red),'finger_contacts':contacts,'lifted':self.max_height>.13,'placed':math.dist(red,self.goal)<.055 and abs(red[2]-.025)<.015,'completed':self.completed}}

    def render(self):
        with self.lock:
            if self._render_cache is not None and time.monotonic()-self._render_at<.5:return self._render_cache
            view=p.computeViewMatrixFromYawPitchRoll([.35,.02,.3],1.3,48,-28,0,2)
            proj=p.computeProjectionMatrixFOV(48,1.5,.05,3)
            pixels=self.call(p.getCameraImage,600,400,viewMatrix=view,projectionMatrix=proj,renderer=p.ER_TINY_RENDERER,shadow=1)[2]
            out=BytesIO();Image.fromarray(np.asarray(pixels,dtype=np.uint8).reshape(400,600,4)[:,:,:3]).save(out,format='JPEG',quality=85)
            self._render_cache=out.getvalue();self._render_at=time.monotonic();return self._render_cache

    def close(self):
        if p.isConnected(self.cid):p.disconnect(self.cid)
