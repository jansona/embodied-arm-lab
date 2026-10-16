"""Bounded RGB waypoint execution for SO101, with explicit model provenance.

This is a local visual waypoint experiment, not a full pick/stack planner.
The simulator owns the physics tick loop. No evaluation/oracle method is used.
"""
import math
import threading
import time
from embodied_arm.control.so101_control import robot_only, MAX_POSITION_STD_M

class VisualWaypointExecutor:
    def __init__(self,simulation,scene_source,controller,*,clock=time.monotonic,sleep=time.sleep):
        self.sim=simulation;self.source=scene_source;self.controller=controller
        self.clock=clock;self.sleep=sleep;self.lock=threading.RLock();self.generation=0
        self.result={'status':'idle','calls':0,'actions':0,'controller':'jev','source':'rgb_only'}
    def status(self):
        with self.lock:return dict(self.result)
    def stop(self):
        with self.lock:self.generation+=1
        self.source.cancel()
        if hasattr(self.controller,'cancel'):self.controller.cancel()
        self.sim.stop()
    def run(self,object_id,offset_xyz,*,max_calls=12,max_seconds=90):
        if object_id not in ('red','blue'):raise ValueError('Unknown visual landmark')
        if (len(offset_xyz)!=3 or any(type(x) not in (float,int) or not math.isfinite(x) or abs(x)>.06 for x in offset_xyz)
                or not 1<=max_calls<=12 or not 1<=max_seconds<=180):raise ValueError('Invalid bounded waypoint')
        with self.lock:
            if self.result['status']=='running':raise ValueError('Already running')
            self.generation+=1;generation=self.generation
            self.result={'status':'running','calls':0,'actions':0,'controller':'jev','source':'rgb_only','object_id':object_id}
        deadline=self.clock()+max_seconds
        try:
            self.source.refresh(required_objects=('red','blue')) # One semantic Qwen observation per explicit trial.
            for index in range(max_calls+1):
                self._check(generation,deadline)
                snapshot=self.source.track() # Fresh RGB even before the first move.
                self._check(generation,deadline)
                from embodied_arm.perception.camera_perception import EstimatedScene
                scene=EstimatedScene.from_dict(snapshot['scene'])
                point=scene.require(object_id,now=self.clock(),max_age_s=30)
                if point.position_std_m>MAX_POSITION_STD_M:raise ValueError('uncertain_landmark')
                robot=robot_only(snapshot['robot']);target=[a+b for a,b in zip(point.xyz,offset_xyz)]
                if any(not lo<=x<=hi for x,(lo,hi) in zip(target,robot['workspace'])):raise ValueError('target_outside_workspace')
                obstacle=scene.require('blue' if object_id=='red' else 'red',now=self.clock(),max_age_s=3)
                obstacle_center=[*obstacle.xyz[:2],obstacle.xyz[2]-.01]
                self._clearance(robot['ee'],target,obstacle_center)
                from embodied_arm.perception.so101_uncertainty import axis_bands
                bands=axis_bands([a-b for a,b in zip(target,robot['ee'])],point.covariance_m2)
                self.result['axis_uncertainty']=bands
                if all(item['status']=='inside' for item in bands):
                    self.result.update(status='completed',reason='estimated_uncertainty_envelope_within_tolerance')
                    return self.status()
                if not any(item['status']=='outside' for item in bands):raise ValueError('uncertainty_unresolved')
                if self.result['calls']>=max_calls:break
                # Fixed X, then Y, then Z subplanning; no direction is selected here.
                axis=next(i for i,item in enumerate(bands) if item['status']=='outside')
                from embodied_arm.providers.so101_jev import build_observation
                request=build_observation({'source':'rgb_only','kind':'cartesian','target_xyz':target,'tolerance_m':.003,'step_m':.008},robot,axis)
                self.result['calls']+=1
                decision=self.controller.decide(request)
                self._check(generation,deadline)
                fresh=robot_only(self.sim.proprioception())
                if (fresh['epoch']!=robot['epoch'] or fresh['actions']!=robot['actions'] or fresh['busy'] or not fresh['paused']
                        or math.dist(fresh['ee'],robot['ee'])>.001):raise ValueError('robot_changed_during_decision')
                self.source.snapshot() # Also enforces perception age after model latency.
                if not decision.accepted:raise ValueError('controller_rejected')
                self._apply(decision,robot,target,generation,deadline,axis,bands[axis]['margin_m'],obstacle_center)
            self.result.update(status='stopped',reason='call_budget_exhausted')
        except Exception as exc:
            allowed={'uncertain_landmark','target_outside_workspace','robot_changed_during_decision','controller_rejected','cancelled','deadline','motion_timeout','unsafe_hover_path','uncertainty_unresolved'}
            self.result.update(status='stopped',reason=str(exc) if str(exc) in allowed else 'perception_or_controller_error')
        finally:
            self.sim.stop()
            if self.result['status']=='running':self.result.update(status='stopped',reason='interrupted')
        return self.status()
    @staticmethod
    def _clearance(start,end,obstacle):
        # Conservative local TCP envelope; this is not full-arm collision planning.
        # Both block observations are required; no unknown obstacle is treated as empty.
        if min(start[2],end[2])<.045:raise ValueError('unsafe_hover_path')
        delta=[b-a for a,b in zip(start,end)];length=sum(x*x for x in delta)
        fraction=0 if not length else max(0,min(1,sum((c-a)*d for c,a,d in zip(obstacle,start,delta))/length))
        nearest=[a+fraction*d for a,d in zip(start,delta)]
        if math.dist(nearest,obstacle)<.08:raise ValueError('unsafe_hover_path')
    def _check(self,generation,deadline):
        if generation!=self.generation:raise ValueError('cancelled')
        if self.clock()>=deadline:raise ValueError('deadline')
    def _apply(self,decision,robot,target,generation,deadline,scheduled_axis,margin,obstacle_center):
        action=decision.action
        if action=='hold':return
        allowed={f'{axis}_{sign}' for axis in 'xyz' for sign in ('plus','minus')}
        if action not in allowed:raise ValueError('controller_rejected')
        index='xyz'.index(action[0])
        if index!=scheduled_axis:raise ValueError('controller_rejected')
        sign=1 if action.endswith('plus') else -1
        if sign*(target[index]-robot['ee'][index])<=.003:raise ValueError('controller_rejected')
        xyz=list(robot['ee']);xyz[index]+=sign*min(.008,max(0,abs(target[index]-xyz[index])-margin))
        self._clearance(robot['ee'],xyz,obstacle_center)
        self._check(generation,deadline)
        self.sim.resume()
        self.sim.manual('xyz',xyz,origin='jev')
        self.result['actions']+=1
        motion_deadline=min(deadline,self.clock()+5.)
        while self.sim.proprioception()['busy']:
            self._check(generation,deadline)
            if self.clock()>=motion_deadline:raise ValueError('motion_timeout')
            self.sleep(.02)
        self.sim.pause_idle()
