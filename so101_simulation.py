"""SO-ARM101 physics with RGB cameras and a separate evaluation-only oracle.

Model/control code must use proprioception() and capture_frames(). Body poses,
object contact IDs and evaluation scores are deliberately absent from both.
"""
from __future__ import annotations
import base64
from collections import deque
from io import BytesIO
import math
import threading
import time
import uuid
import numpy as np
from PIL import Image
import pybullet as p
import pybullet_data

WORKSPACE=((.07,.33),(-.19,.19),(.008,.33))
CUBE_EDGE=.02
CAMERAS={'left':((.08,-.46,.40),(.20,0,.035)),
         'right':((.44,.30,.39),(.20,0,.035))}


def camera_calibration(camera_id,width=640,height=480):
    eye,target=map(lambda v:np.asarray(v,dtype=float),CAMERAS[camera_id])
    forward=(target-eye)/np.linalg.norm(target-eye)
    right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    matrix=np.eye(4);matrix[:3,:3]=np.column_stack((right,down,forward));matrix[:3,3]=eye
    focal=height/(2*math.tan(math.radians(50)/2))
    return {'camera_id':camera_id,'width':width,'height':height,'fx':focal,'fy':focal,
            'cx':(width-1)/2,'cy':(height-1)/2,'world_from_camera':matrix.tolist()}


class SO101Simulation:
    def __init__(self,seed=0,*,camera_size=(640,480),illumination_profile="standard"):
        if illumination_profile not in ("standard","shadow_free_rgb_v1"):raise ValueError("Unsupported illumination profile")
        self.illumination_profile=illumination_profile
        if tuple(camera_size) not in ((640,480),(1280,960),(1920,1440)):raise ValueError('Unsupported stereo resolution')
        self.camera_size=tuple(camera_size)
        self.lock=threading.RLock();self.cid=p.connect(p.DIRECT);self.epoch=0
        self.reset(seed)
    def call(self,fn,*args,**kwargs):return fn(*args,physicsClientId=self.cid,**kwargs)
    def reset(self,seed=0):
        from so101_robot import SO101Robot
        with self.lock:
            self.epoch+=1;self.seed=int(seed);rng=np.random.default_rng(seed)
            self.call(p.resetSimulation);self.call(p.setAdditionalSearchPath,pybullet_data.getDataPath())
            self.call(p.setGravity,0,0,-9.81);self.call(p.setTimeStep,1/240)
            self.call(p.setPhysicsEngineParameter,numSolverIterations=150,deterministicOverlappingPairs=1)
            self.call(p.loadURDF,'plane.urdf');self.robot=SO101Robot(self.cid)
            self._objects={}
            jitter=lambda:float(rng.uniform(-.006,.006)) if seed else 0.
            for name,xyz,color in [('red',[.19+jitter(),-.05+jitter(),.01],[.94,.12,.12,1]),('blue',[.25+jitter(),.055+jitter(),.01],[.12,.30,.95,1])]:
                collision=self.call(p.createCollisionShape,p.GEOM_BOX,halfExtents=[CUBE_EDGE/2]*3)
                visual=self.call(p.createVisualShape,p.GEOM_BOX,halfExtents=[CUBE_EDGE/2]*3,rgbaColor=color)
                body=self.call(p.createMultiBody,baseMass=.012,baseCollisionShapeIndex=collision,baseVisualShapeIndex=visual,basePosition=xyz)
                self.call(p.changeDynamics,body,-1,lateralFriction=1.2,spinningFriction=.01,rollingFriction=.001)
                self._objects[name]=body
            self._goal=[.17,.105,.001]
            marker=self.call(p.createVisualShape,p.GEOM_BOX,halfExtents=[.025,.025,.0005],rgbaColor=[.08,.82,.30,1])
            self.call(p.createMultiBody,baseMass=0,baseVisualShapeIndex=marker,basePosition=self._goal)
            self.queue=deque();self.current=None;self.paused=True;self.actions=0;self.steps=0
            self.status='Ready; camera perception required';self.error=None;self.origin='reset'
            self._renders={};self.robot.reset_home();self.robot.apply_motors()
            for _ in range(120):self.call(p.stepSimulation)
    def proprioception(self):
        with self.lock:
            safe=dict(self.robot.proprioception())
            # Default SO101 has no force sensors; simulated contacts are oracle-only.
            safe.pop('gripper_tactile',None);safe.pop('tactile_force_unit',None)
            # Explicitly limited to robot encoders/FK and runtime state. No object
            # contact identities, simulator masks, camera depth or task metrics.
            safe.update(schema='so101-proprioception-v1',simulation_only=True,epoch=self.epoch,
                        paused=self.paused,busy=bool(self.current or self.queue),actions=self.actions,
                        physics_steps=self.steps,status=self.status,error=self.error,origin=self.origin,
                        workspace=[list(x) for x in WORKSPACE],cube_edge_m=CUBE_EDGE)
            return safe
    def _rgb(self,camera_id,roi=None):
        eye,target=CAMERAS[camera_id];width,height=self.camera_size
        view=p.computeViewMatrix(eye,target,[0,0,1]);proj=p.computeProjectionMatrixFOV(50,width/height,.01,2.)
        native_width,native_height=width,height
        if roi is not None:
            from rgb_roi_geometry import frustum
            proj=p.computeProjectionMatrix(*frustum(roi,width,height));width=roi[2]-roi[0];height=roi[3]-roi[1]
        pixels=self.call(p.getCameraImage,width,height,viewMatrix=view,projectionMatrix=proj,
                         renderer=p.ER_TINY_RENDERER,flags=p.ER_NO_SEGMENTATION_MASK,shadow=int(self.illumination_profile=="standard"))[2]
        rgb=np.asarray(pixels,dtype=np.uint8).reshape(height,width,4)[:,:,:3]
        if roi is not None:
            canvas=np.zeros((native_height,native_width,3),dtype=np.uint8)
            canvas[roi[1]:roi[3],roi[0]:roi[2]]=rgb;rgb=canvas
        out=BytesIO();Image.fromarray(rgb).save(out,format='JPEG',quality=92)
        return out.getvalue()
    def capture_frames(self,*,roi_bounds=None):
        with self.lock:
            if roi_bounds is not None and set(roi_bounds)!=set(CAMERAS):raise ValueError('Both stereo crop bounds required')
            frame_id=uuid.uuid4().hex;captured_at=time.monotonic();images=[]
            for camera_id in CAMERAS:
                jpeg=self._rgb(camera_id,None if roi_bounds is None else roi_bounds[camera_id])
                images.append({'camera_id':camera_id,'width':self.camera_size[0],'height':self.camera_size[1],
                               'data_url':'data:image/jpeg;base64,'+base64.b64encode(jpeg).decode('ascii')})
                if roi_bounds is None:self._renders[camera_id]=(time.monotonic(),jpeg)
            packet={'frame_id':frame_id,'captured_at':captured_at,'images':images,
                    'calibrations':[camera_calibration(k,*self.camera_size) for k in CAMERAS]}
            if roi_bounds is not None:packet.update(capture_profile='native_coordinate_roi_canvas',observed_roi_bounds=roi_bounds)
            return packet
    def render(self,camera_id='left'):
        if camera_id not in CAMERAS:raise ValueError('Unknown RGB camera')
        with self.lock:
            old=self._renders.get(camera_id)
            if old and time.monotonic()-old[0]<.4:return old[1]
            rgb=self._rgb(camera_id);self._renders[camera_id]=(time.monotonic(),rgb);return rgb
    def manual(self,action,values=None,*,origin="manual"):
        with self.lock:
            if origin not in ('manual','jev','offline_fixture'):raise ValueError('Invalid command provenance')
            if self.paused or self.current or self.queue:raise ValueError('Resume an idle robot before moving')
            if action=='xyz':
                xyz=list(values) if isinstance(values,(list,tuple)) else []
                if len(xyz)!=3 or not all(type(v) in (int,float) and math.isfinite(v) and lo<=v<=hi for v,(lo,hi) in zip(xyz,WORKSPACE)):raise ValueError('Target outside SO101 workspace')
                current=self.robot.tcp_pose()[0]
                if math.dist(current,xyz)>.020001:raise ValueError('SO101 Cartesian command limited to2cm')
                goal=list(self.robot.solve_ik(xyz))
            elif action=='joints':
                goal=list(values) if isinstance(values,(list,tuple)) else []
                if len(goal)!=5 or max(abs(a-b) for a,b in zip(goal,self.robot.joint_positions()))>.150001:raise ValueError('Joint jog exceeds0.15rad')
                if not all(math.isfinite(q) and lo<=q<=hi for q,lo,hi in zip(goal,self.robot.lower_limits,self.robot.upper_limits)):raise ValueError('Joint limit')
            elif action in ('open','close'):goal=None
            else:raise ValueError('Unsupported SO101 command')
            self.queue.append({'joint_goal':goal,'gripper':1. if action=='open' else 0. if action=='close' else None,
                               'ticks':360 if action in ('open','close') else 144,
                               'settle_ticks':120 if action=='xyz' else 0,'elapsed':0,'origin':origin})
            self.actions+=1
    def resume(self):
        with self.lock:self.paused=False;self.status='Ready'
    def pause_idle(self):
        """Freeze only an already completed motion for the paused RGB profile."""
        with self.lock:
            if self.current or self.queue:raise ValueError("Cannot observe during motion")
            self.paused=True
    def stop(self):
        with self.lock:
            # Cancel an active gripper trajectory at its measured position.
            # An idle/arm-only stop must retain a completed gripping setpoint:
            # replacing it with the contact-limited encoder angle removes force.
            freeze_gripper=bool(self.current and self.current.get('gripper') is not None)
            self.epoch+=1;self.queue.clear();self.current=None;self.paused=True;self.status='Stopped'
            self.robot.command_joints(self.robot.joint_positions())
            if freeze_gripper:
                angle=self.call(p.getJointState,self.robot.body_id,self.robot.gripper_joint_index)[0]
                lo,hi=self.robot.gripper_limits
                self.robot.command_gripper(max(0.,min(1.,(angle-lo)/(hi-lo))))
            self.robot.apply_motors()
    def tick(self,count=4):
        with self.lock:
            if self.paused:return
            for _ in range(count):
                if self.current is None and self.queue:
                    self.current=self.queue.popleft();self.current['start']=list(self.robot.joint_positions())
                    if self.current['gripper'] is not None:
                        angle=self.call(p.getJointState,self.robot.body_id,self.robot.gripper_joint_index)[0]
                        lo,hi=self.robot.gripper_limits
                        self.current['gripper_start']=max(0.,min(1.,(angle-lo)/(hi-lo)))
                    self.status='Moving';self.origin=self.current['origin']
                if self.current:
                    c=self.current;c['elapsed']+=1;t=min(1,c['elapsed']/c['ticks']);t=t*t*(3-2*t)
                    if c['joint_goal'] is not None:self.robot.command_joints([a+t*(b-a) for a,b in zip(c['start'],c['joint_goal'])])
                    if c['gripper'] is not None:
                        self.robot.command_gripper(c['gripper_start']+t*(c['gripper']-c['gripper_start']))
                    # Bounded 0.5s Cartesian servo dwell; app still owns physics.
                    # Keep the same accepted target, never add a direction/action.
                    if c['elapsed']>=c['ticks']+c.get('settle_ticks',0):self.current=None;self.status='Motion complete'
                self.robot.apply_motors();self.call(p.stepSimulation);self.steps+=1
    def grasp_geometry(self,target_xyz=None):
        """Known jaw geometry + encoder FK, not an object/environment observation.

        The horizontal approach offset avoids the fixed fingertip during descent.
        Its sampled capture window is documented in SO101-GRASP-FEASIBILITY.md.
        """
        with self.lock:
            joints=self.robot.joint_positions() if target_xyz is None else self.robot.solve_ik(target_xyz)
            transform=self.robot.forward_kinematics(joints)
            axis=[float(transform[0,0]),float(transform[1,0]),0.]
            length=math.hypot(axis[0],axis[1])
            if length<.9:raise ValueError('Top-down jaw orientation required')
            shift=math.sqrt(2)*.01+.006+.002-.012
            return {'approach_offset_world':[shift*x/length for x in axis],
                    'tcp_below_top_m':.007,'transit_height_m':.06,
                    'profile':'planar_fixed_jaw_clearance_v1'}
    def evaluation_state(self):
        """Oracle ONLY for independent scoring/debug. Never a controller input."""
        with self.lock:
            objects={name:list(self.call(p.getBasePositionAndOrientation,body)[0]) for name,body in self._objects.items()}
            return {'schema':'evaluation-oracle-v1','not_for_model_input':True,'ground_truth_objects':objects,
                    'ground_truth_goal':list(self._goal),'seed':self.seed,'cube_edge_m':CUBE_EDGE,
                    'cube_to_cube_contacts':len(self.call(p.getContactPoints,self._objects['red'],self._objects['blue']))}
    def close(self):
        if p.isConnected(self.cid):p.disconnect(self.cid)
