"""Single-object pick/place recipes grounded exclusively in image observations.

The phase observer must validate fresh RGB and dynamic identity, and the driver
must obtain every direction/gripper decision from Jev. Neither dependency may
substitute simulator object poses. Explicit experimental mode is restricted to the exact local simulator; runtime
visual and movement guards remain mandatory.
"""
import math
import threading
import time
from embodied_arm.perception.camera_perception import EstimatedScene
from embodied_arm.control.so101_control import robot_only

class MotionStopped(ValueError):
    pass

class VisualManipulationExecutor:
    supports_cycles = True
    supported_targets = ["goal","on:red","on:blue"]

    def __init__(self, *, simulation, observer, driver, clock=time.monotonic, max_calls=120, max_seconds=300, experimental_simulation=False):
        if not 1 <= max_calls <= 200 or not 1 <= max_seconds <= 420:
            raise ValueError('invalid_budget')
        if experimental_simulation:
            from embodied_arm.simulation.so101_simulation import SO101Simulation
            if type(simulation) is not SO101Simulation:raise ValueError("simulation_only")
        self.experimental_simulation=bool(experimental_simulation)
        self.sim = simulation
        self.observer = observer
        self.driver = driver
        self.clock = clock
        self.max_calls = max_calls
        self.max_seconds = max_seconds
        self.held = None
        self.held_evidence = None
        self._run_state = None
        self._support_policy = None
        self.validated = bool(getattr(observer, 'validated', False) and getattr(driver, 'validated', False))

    def readiness(self):
        if not (self.validated or self.experimental_simulation):
            return {'available':False,'reason':'visual_execution_not_validated'}
        controller=getattr(self.driver,'controller',None)
        status=getattr(controller,'status',None)
        if status is None:
            return {'available':False,'reason':'controller_readiness_unknown'}
        try:
            state=status()
            available=state.get('available') is True
        except Exception:
            available=False
        return {'available':available,'reason':None if available else 'jev_unavailable'}

    def reset(self):
        self.stop()
        self.held=None
        self.held_evidence=None

    def stop(self):
        self._run_state=None
        self._support_policy=None
        self.held_evidence=None
        self.sim.stop()
        if hasattr(self.observer, 'cancel'):
            self.observer.cancel()

    def planning_held_object(self):
        run=self._run_state
        if run is not None and run.get('holding_state')!='held_verified':return None
        evidence=self.held_evidence
        if evidence and evidence.get("epoch")==self.sim.proprioception().get("epoch"):
            return self.held
        return None

    def pause(self):
        self.sim.pause_idle()

    def bounded_execute(self, plan, *, cancel_event=None, event_callback=None, max_calls, deadline,
                        cycle_max_calls=None, resume_token=None):
        return self.execute(plan,cancel_event=cancel_event,event_callback=event_callback,
                            max_calls=max_calls,deadline=deadline,cycle_max_calls=cycle_max_calls,
                            resume_token=resume_token)

    @staticmethod
    def _intent(plan):
        return {key:plan.get(key) for key in ('action','object_id','target_id')}

    def _continuity(self, previous):
        current=robot_only(self.sim.proprioception())
        if (current['busy'] or not current['paused'] or
            any(current[k]!=previous[k] for k in ('epoch','actions')) or
            math.dist(current['ee'],previous['ee'])>.001 or
            max(abs(a-b) for a,b in zip(current['joints'],previous['joints']))>.005 or
            abs(current['gripper_openness']-previous['gripper_openness'])>.01):
            raise ValueError('resume_scene_changed')
        return current

    def execute(self, plan, *, cancel_event=None, event_callback=None, max_calls=None, deadline=None,
                cycle_max_calls=None, resume_token=None):
        import secrets
        if not (self.validated or self.experimental_simulation):raise ValueError('visual_execution_not_validated')
        intent=self._intent(plan);obj=intent['object_id']
        if intent['action'] not in ('pick','place') or obj not in ('red','blue'):raise ValueError('unsupported_task')
        if intent['action']=='pick' and intent['target_id'] is not None:raise ValueError('unsupported_task')
        if intent['action']=='place' and intent['target_id'] not in ('goal','on:red','on:blue'):raise ValueError('unsupported_task')
        if intent['target_id']=='on:'+obj:raise ValueError('self_placement')
        limit=self.max_calls if max_calls is None else max_calls
        if type(limit) is not int or not 1<=limit<=self.max_calls:raise ValueError('invalid_budget')
        cycle=limit if cycle_max_calls is None else cycle_max_calls
        if type(cycle) is not int or not 1<=cycle<=200:raise ValueError('invalid_budget')
        deadline=min(self.clock()+self.max_seconds,deadline if deadline is not None else float('inf'))
        if not math.isfinite(deadline) or deadline<=self.clock():raise ValueError('time_budget')
        cancel=cancel_event or threading.Event()
        run=getattr(self,'_run_state',None)
        if run is not None:
            if not isinstance(resume_token,str) or resume_token!=run['token'] or intent!=run['intent']:
                raise ValueError('invalid_resume')
            self._continuity(run['robot'])
            if deadline>run['deadline']+1e-6:raise ValueError('resume_deadline_extended')
            deadline=min(deadline,run['deadline'])
            run['token']=None
        elif resume_token is not None:raise ValueError('invalid_resume')
        else:
            if intent['action']=='pick' and self.held is not None:raise ValueError('already_holding')
            if intent['action']=='place' and (self.held!=obj or (self.experimental_simulation and self.planning_held_object()!=obj)):raise ValueError('verified_held_object_required')
            run={'intent':intent,'program':self._program(intent),'pending':None,'token':None,
                 'deadline':deadline,'initial_limit':limit,'total_calls':0,'total_actions':0,
                 'phases':[],'sources':set(),'retries':0,'phase':'approach','holding_state':'held_verified' if self.held else 'empty',
                 'recent_actions':[]}
            self._run_state=run
            self.last_driver_outcome=None
        limit=min(limit,run['initial_limit']-run['total_calls'])
        calls=actions=0;last_snapshot=None
        def checked():
            if cancel.is_set():raise ValueError('cancelled')
            if self.clock()>=deadline:raise ValueError('time_budget')
        def emit(detail):
            checked()
            if event_callback:event_callback(detail)
        def observation():
            nonlocal last_snapshot
            checked()
            try:last_snapshot=self.observer.observe()
            except ValueError:
                if run['retries']>=1:raise
                run['initial_perception_error']=getattr(self.observer,'last_error',None)
                run['retries']+=1
                emit({'phase':'observation_recovery','attempt':run['retries'],'reason':'fresh_rgb_rejected','calls':calls,'actions':actions})
                checked();last_snapshot=self.observer.observe()
            self._point(last_snapshot,obj)
            return last_snapshot
        def result(status,reason,**extra):
            return {'status':status,'verified':status=='completed','reason':reason,'action':intent['action'],
                'object_id':obj,'calls':calls,'actions':actions,'phase':run['phase'],'phases':list(run['phases']),
                'holding_state':run['holding_state'],'source':'rgb_only','controller':'jev',
                'decision_provenance':sorted(run['sources']),'observation_retries':run['retries'],
                'recent_actions':list(run['recent_actions']),**extra}
        def suspend():
            checked();self.sim.pause_idle()
            run['robot']=robot_only(self.sim.proprioception());run['token']=secrets.token_hex(16)
            return result('yielded','cycle_budget',resume_token=run['token'])
        try:
            checked()
            while True:
                op=run['pending']
                if op is None:
                    try:op=next(run['program']);run['pending']=op
                    except StopIteration as complete:
                        feedback=complete.value
                        if not isinstance(feedback,dict) or feedback.get('verified') is not True:raise ValueError('visual_verification_failed')
                        checked();self.sim.pause_idle();self._run_state=None;self._support_policy=None
                        return result('completed','visually_verified',feedback=feedback)
                run['phase']=op['phase']
                if not run['phases'] or run['phases'][-1]!=op['phase']:
                    run['phases'].append(op['phase']);emit({'phase':op['phase'],'calls':calls,'actions':actions})
                if op['kind']=='observe':
                    checked();last_snapshot=self.observer.begin_phase(op['phase'],obj);self._point(last_snapshot,obj)
                    response=last_snapshot
                else:
                    if calls>=limit:raise ValueError('call_budget_exhausted')
                    if calls>=cycle:return suspend()
                    target=op['target'];active=None
                    def fresh():
                        snapshot=observation()
                        if op['kind']=='gripper' and target==0:self._check_grasp_alignment(snapshot,obj,op['geometry'])
                        if op['kind']=='gripper' and op.get('phase')=='release':
                            self._check_release_alignment(snapshot,obj,op['release_resolver'])
                        if op['kind']=='move' and 'resolver' not in op:self._check_path(snapshot,obj,target)
                        return snapshot
                    def update(snapshot):
                        point=self._point(snapshot,obj)
                        resolved=op['resolver'](snapshot,point)
                        next_target=resolved['target_xyz'] if isinstance(resolved,dict) else resolved
                        covariance=resolved['covariance_m2'] if isinstance(resolved,dict) else point.covariance_m2
                        self._check_path(snapshot,obj,next_target)
                        return {'target_xyz':list(next_target),'covariance_m2':covariance}
                    def micro_event(detail):
                        if detail.get('phase') in ('motion_applied','motion_completed') and run['recent_actions']:
                            run['recent_actions'][-1]['applied']=True
                        if detail.get('phase')=='jev_decision':
                            run['recent_actions'].append({'action':detail.get('action') or 'hold','applied':False})
                            run['recent_actions']=run['recent_actions'][-8:]
                        emit({**detail,'calls':calls+detail.get('calls',0),'actions':actions+detail.get('actions',0)})
                    allowance=min(limit-calls,cycle-calls)
                    options={'cancel_event':cancel,'max_calls':allowance,'deadline':deadline,
                             'before_decision':fresh,'after_motion':observation,'event_callback':micro_event,
                             'yield_on_budget':cycle_max_calls is not None and allowance<limit-calls}
                    if op['kind']=='move':
                        if 'resolver' in op:options['target_update']=update
                        outcome=self.driver.move_to(target,op['covariance'],**options)
                    else:outcome=self.driver.gripper(target,**options)
                    used=outcome.get('calls');applied=outcome.get('actions')
                    if type(used) is not int or not 0<=used<=allowance or type(applied) is not int or applied<0:raise ValueError('invalid_driver_accounting')
                    calls+=used;actions+=applied;run['total_calls']+=used;run['total_actions']+=applied
                    run['sources'].add(outcome.get('provenance','unverified'));self.last_driver_outcome=outcome
                    checked()
                    if outcome.get('status')=='yielded':return suspend()
                    if outcome.get('status') not in ('completed','actuation_completed'):raise MotionStopped(outcome.get('reason','microcontrol_stopped'))
                    if op['phase']=='close':run['holding_state']='unknown'
                    if op['phase']=='release':run['holding_state']='release_pending'
                    response=outcome
                checked()
                try:run['pending']=run['program'].send(response)
                except StopIteration as complete:
                    feedback=complete.value
                    if not isinstance(feedback,dict) or feedback.get('verified') is not True:raise ValueError('visual_verification_failed')
                    checked();self.sim.pause_idle();self._run_state=None;self._support_policy=None
                    return result('completed','visually_verified',feedback=feedback)
        except Exception as exc:
            allowed={'cancelled','time_budget','call_budget_exhausted','precision_floor','perception_rejected','controller_rejected',
                     'uncertainty_unresolved','grasp_not_visually_verified','release_not_visually_verified','unknown_support',
                     'unsafe_release_geometry','prerelease_alignment_unverified','prerelease_object_alignment_unverified','preclose_alignment_unverified','horizontal_motion_below_transit_height','observed_obstacle_clearance'}
            reason=str(exc) if str(exc) in allowed else 'execution_failed'
            self._run_state=None;self._support_policy=None;self.sim.stop()
            if run['holding_state']!='empty':run['holding_state']='unknown'
            return result('cancelled' if cancel.is_set() else 'failed',reason,
                          perception_detail=run.get('initial_perception_error') or getattr(self.observer,'last_error',None),
                          recovery_detail=getattr(self.observer,'last_error',None),driver=getattr(self,'last_driver_outcome',None))

    def _program(self,plan):
        obj=plan['object_id']
        self.observer.configure_task(plan['action'])
        if hasattr(self.observer,'begin_step'):
            support=plan['target_id']
            if isinstance(support,str) and support.startswith('on:'):support=support.split(':',1)[1]
            self.observer.begin_step(plan['action'],obj,target_id=support,held_evidence=getattr(self,'held_evidence',None))
        before=yield {'kind':'observe','phase':'approach'}
        point=self._point(before,obj);center=[*point.xyz[:2],point.xyz[2]-.01]
        geometry=self.sim.grasp_geometry([center[0],center[1],point.xyz[2]-.007]);offset=geometry['approach_offset_world']
        def move(phase,target,covariance,resolver=None):
            op={'kind':'move','phase':phase,'target':target,'covariance':covariance}
            if resolver:op['resolver']=resolver
            return op
        if plan['action']=='pick':
            yield {'kind':'gripper','phase':'approach','target':1}
            yield move('approach',[center[0]+offset[0],center[1]+offset[1],.06],point.covariance_m2,
                       lambda snapshot,p:[p.xyz[0]+offset[0],p.xyz[1]+offset[1],.06])
            descend=yield {'kind':'observe','phase':'descend'};point=self._point(descend,obj)
            yield move('descend',[point.xyz[0]+offset[0],point.xyz[1]+offset[1],point.xyz[2]-geometry['tcp_below_top_m']],point.covariance_m2,
                       lambda snapshot,p:[p.xyz[0]+offset[0],p.xyz[1]+offset[1],p.xyz[2]-geometry['tcp_below_top_m']])
            yield {'kind':'observe','phase':'preclose'}
            self.observer.begin_phase('closing',obj)
            yield {'kind':'gripper','phase':'close','target':0,'geometry':geometry}
            baseline=yield {'kind':'observe','phase':'lift'}
            robot=robot_only(baseline['robot']);point=self._point(baseline,obj)
            yield move('lift',[robot['ee'][0],robot['ee'][1],.06],point.covariance_m2)
            lifted=yield {'kind':'observe','phase':'verify_lift'}
            feedback=self.observer.verify_lift(baseline,lifted,obj)
            if feedback.get('verified') is not True:raise ValueError('grasp_not_visually_verified')
            self.held=obj;self.held_evidence={'object_id':obj,'epoch':lifted['robot']['epoch'],'actions':lifted['robot']['actions'],
                'frame_ids':lifted['scene']['frame_ids'],'identity_token':lifted['visual_evidence']['identity_token'],
                'verified_lift':True,'source':'phase_scoped_temporal_rgb'}
            self._run_state['holding_state']='held_verified'
            return feedback
        from embodied_arm.perception.camera_perception import ObjectEstimate
        support_id=plan['target_id'].split(':',1)[1] if plan['target_id'].startswith('on:') else None
        goal=self._point(before,support_id) if support_id else ObjectEstimate.from_dict(self.observer.fixed_goal()['estimate'])
        robot=robot_only(before['robot']);grip_offset=[a-b for a,b in zip(robot['ee'],center)]
        release=[goal.xyz[0]+grip_offset[0],goal.xyz[1]+grip_offset[1],goal.xyz[2]+.01+grip_offset[2]+.002]
        maximum=.07 if support_id else .04
        if not .008<=release[2]<=maximum:raise ValueError('unsafe_release_geometry')
        transit=max(.06,release[2]+.03)
        if transit>.10:raise ValueError('unsafe_release_geometry')
        self._support_policy={'support_id':support_id,'grip_offset':grip_offset} if support_id else None
        def placement_target(snapshot,carried,lower=False):
            destination=self._point(snapshot,support_id) if support_id else ObjectEstimate.from_dict(self.observer.fixed_goal()['estimate'])
            covariance=[[destination.covariance_m2[i][j]+carried.covariance_m2[i][j] for j in range(3)] for i in range(3)]
            target=[destination.xyz[0]+grip_offset[0],destination.xyz[1]+grip_offset[1],
                    destination.xyz[2]+.01+grip_offset[2]+.002 if lower else transit]
            if lower and not .008<=target[2]<=maximum:raise ValueError('unsafe_release_geometry')
            return {'target_xyz':target,'covariance_m2':covariance,
                    'expected_object_top_xyz':[destination.xyz[0],destination.xyz[1],destination.xyz[2]+.022]}
        combined=[[goal.covariance_m2[i][j]+point.covariance_m2[i][j] for j in range(3)] for i in range(3)]
        yield move('transit',[release[0],release[1],transit],combined,placement_target)
        yield {'kind':'observe','phase':'lower'}
        yield move('lower',release,combined,lambda snapshot,p:placement_target(snapshot,p,True))
        yield {'kind':'gripper','phase':'release','target':1,
               'release_resolver':lambda snapshot,p:placement_target(snapshot,p,True)}
        released=yield {'kind':'observe','phase':'released'}
        released_robot=robot_only(released['robot'])
        yield move('retreat',[released_robot['ee'][0],released_robot['ee'][1],transit],combined)
        retreated=yield {'kind':'observe','phase':'verify_release'}
        if support_id:
            from embodied_arm.control.stack_verification import verify_stack_release
            feedback=verify_stack_release(before,released,retreated,obj,support_id,now=self.clock(),lease_seconds=getattr(self.observer,'lease_seconds',120))
        else:feedback=self.observer.verify_release(before,released,retreated,obj,list(goal.xyz))
        if feedback.get('verified') is not True:raise ValueError('release_not_visually_verified')
        self.held=None;self.held_evidence=None;self._run_state['holding_state']='empty'
        return feedback

    def _check_release_alignment(self, snapshot, object_id, resolver):
        from embodied_arm.perception.so101_uncertainty import axis_bands
        point=self._point(snapshot,object_id)
        resolved=resolver(snapshot,point)
        robot=robot_only(snapshot['robot'])
        self._check_path(snapshot,object_id,resolved['target_xyz'])
        bands=axis_bands([a-b for a,b in zip(resolved['target_xyz'],robot['ee'])],resolved['covariance_m2'])
        if not all(b['status']=='inside' for b in bands):
            raise ValueError('prerelease_alignment_unverified')
        expected=resolved.get('expected_object_top_xyz')
        if expected is None:raise ValueError('prerelease_object_alignment_unverified')
        object_bands=axis_bands([a-b for a,b in zip(expected,point.xyz)],resolved['covariance_m2'])
        if not all(b['status']=='inside' for b in object_bands):
            raise ValueError('prerelease_object_alignment_unverified')

    def _check_grasp_alignment(self, snapshot, object_id, geometry):
        from embodied_arm.perception.so101_uncertainty import axis_bands
        point=self._point(snapshot,object_id)
        robot=robot_only(snapshot['robot'])
        offset=geometry['approach_offset_world']
        target=[point.xyz[0]+offset[0],point.xyz[1]+offset[1],point.xyz[2]-geometry['tcp_below_top_m']]
        bands=axis_bands([a-b for a,b in zip(target,robot['ee'])],point.covariance_m2)
        if not all(b['status']=='inside' for b in bands):
            raise ValueError('preclose_alignment_unverified')

    def _point(self, snapshot, object_id):
        if snapshot.get('source') != 'rgb_only' or snapshot.get('schema') != 'so101-camera-observation-v1':
            raise ValueError('rgb_provenance_required')
        robot = robot_only(snapshot['robot'])
        if robot['busy'] or not robot['paused']:
            raise ValueError('observation_during_motion')
        scene = EstimatedScene.from_dict(snapshot['scene'])
        point = scene.require(object_id, now=self.clock(), max_age_s=3)
        if point.position_std_m > 0.003:
            raise ValueError('visual_uncertainty')
        return point

    def _check_path(self, snapshot, object_id, target):
        robot = robot_only(snapshot['robot'])
        scene = EstimatedScene.from_dict(snapshot['scene'])
        other = scene.require('blue' if object_id == 'red' else 'red', now=self.clock(), max_age_s=3)
        if other.position_std_m > 0.003:
            raise ValueError('unknown_obstacle')
        start = robot['ee']
        delta = [b - a for a, b in zip(start, target)]
        if min(start[2], target[2]) < 0.007:
            raise ValueError('table_clearance')
        if math.hypot(delta[0], delta[1]) > 0.003 and min(start[2], target[2]) < 0.05:
            raise ValueError('horizontal_motion_below_transit_height')
        policy=self._support_policy
        other_id='blue' if object_id=='red' else 'red'
        if policy and policy['support_id']==other_id:
            offset=policy['grip_offset']
            bottom_start=start[2]-offset[2]-.01;bottom_target=target[2]-offset[2]-.01
            clearance=3*other.position_std_m
            lateral=math.hypot(delta[0],delta[1])
            if lateral>.003:
                if min(bottom_start,bottom_target)<other.xyz[2]+.025+clearance:raise ValueError('support_transit_clearance')
            else:
                if math.hypot(target[0]-offset[0]-other.xyz[0],target[1]-offset[1]-other.xyz[1])+clearance>.007:raise ValueError('support_alignment_unverified')
                if min(bottom_start,bottom_target)<other.xyz[2]-.001:raise ValueError('support_surface_clearance')
            return
        center = [*other.xyz[:2], other.xyz[2] - 0.01]
        length = sum((d * d for d in delta))
        t = 0 if not length else max(0, min(1, sum(((p - a) * d for p, a, d in zip(center, start, delta))) / length))
        nearest = [a + t * d for a, d in zip(start, delta)]
        if math.dist(nearest, center) < 0.04 + 3 * other.position_std_m:
            raise ValueError('observed_obstacle_clearance')


def simulation_task_executor(simulation, source, controller):
    """Explicit experimental capability for this exact local PyBullet simulator."""
    from embodied_arm.perception.phase_observer import PhaseSceneObserver
    from embodied_arm.control.jev_motion import JevMotionDriver
    from embodied_arm.services.sequence_execution import SequenceExecutionService
    primitive=VisualManipulationExecutor(simulation=simulation,observer=PhaseSceneObserver(source),
        driver=JevMotionDriver(simulation,controller),experimental_simulation=True,max_calls=200,max_seconds=420)
    return SequenceExecutionService(primitive)
