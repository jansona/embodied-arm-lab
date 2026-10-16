"""Single-object pick/place recipes grounded exclusively in image observations.

The phase observer must validate fresh RGB and dynamic identity, and the driver
must obtain every direction/gripper decision from Jev. Neither dependency may
substitute simulator object poses. Capability is unavailable until both concrete
implementations have passed integration validation.
"""
import math
import threading
import time
from embodied_arm.perception.camera_perception import EstimatedScene
from embodied_arm.control.so101_control import robot_only

class MotionStopped(ValueError):
    pass

class VisualManipulationExecutor:

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

    def stop(self):
        self.sim.stop()
        if hasattr(self.observer, 'cancel'):
            self.observer.cancel()

    def bounded_execute(self, plan, *, cancel_event=None, event_callback=None, max_calls, deadline):
        return self.execute(plan, cancel_event=cancel_event, event_callback=event_callback,
                            max_calls=max_calls, deadline=deadline)

    def execute(self, plan, *, cancel_event=None, event_callback=None, max_calls=None, deadline=None):
        if not (self.validated or self.experimental_simulation):
            raise ValueError('visual_execution_not_validated')
        if plan.get('action') not in ('pick', 'place') or plan.get('object_id') not in ('red', 'blue'):
            raise ValueError('unsupported_task')
        if plan['action'] == 'place' and (plan.get('target_id') != 'goal' or self.held != plan['object_id']):
            raise ValueError('verified_held_object_required')
        cancel = cancel_event or threading.Event()
        call_limit = min(self.max_calls, self.max_calls if max_calls is None else max_calls)
        if type(call_limit) is not int or call_limit < 1:
            raise ValueError("call_budget_exhausted")
        deadline = min(self.clock() + self.max_seconds, deadline if deadline is not None else float("inf"))
        calls = actions = 0
        obj = plan['object_id']
        events = []
        active_target = None
        active_resolver = None
        decision_sources = set()
        observation_retries = 0

        def emit(phase):
            if cancel.is_set() or self.clock() >= deadline:
                raise ValueError('cancelled_or_expired')
            events.append(phase)
            if event_callback:
                event_callback({'phase': phase, 'calls': calls, 'actions': actions})

        def observation(phase):
            emit(phase)
            result = self.observer.begin_phase(phase, obj)
            self._point(result, obj)
            return result

        def checked_fresh():
            nonlocal observation_retries
            try:
                current = self.observer.observe()
            except ValueError:
                # One stationary re-observation can recover a transient RGB failure.
                # It never moves the arm, changes the lease, or accepts missing evidence.
                if observation_retries >= 1 or cancel.is_set() or self.clock() >= deadline:
                    raise
                observation_retries += 1
                if event_callback:event_callback({'phase':'observation_recovery','attempt':observation_retries,
                                                  'reason':'fresh_rgb_rejected','calls':calls,'actions':actions})
                if cancel.is_set() or self.clock() >= deadline:raise ValueError('cancelled_or_expired')
                current = self.observer.observe()
            self._point(current, obj)
            if active_target is not None and active_resolver is None:
                self._check_path(current, obj, active_target)
            return current

        def drive(method, *args, target_resolver=None):
            nonlocal calls, actions, active_target, active_resolver
            active_target = list(args[0]) if method == self.driver.move_to else None
            active_resolver = target_resolver
            if calls >= call_limit:
                raise ValueError('call_budget_exhausted')
            def update(snapshot):
                nonlocal active_target
                point = self._point(snapshot, obj)
                active_target = list(target_resolver(snapshot, point))
                self._check_path(snapshot, obj, active_target)
                return {'target_xyz': active_target, 'covariance_m2': point.covariance_m2}
            def micro_event(detail):
                if event_callback:
                    event_callback({**detail, 'calls': calls + detail.get('calls', 0),
                                    'actions': actions + detail.get('actions', 0)})
            def before_motion():
                snapshot=checked_fresh()
                if method == self.driver.gripper and args[0] == 0:
                    self._check_grasp_alignment(snapshot,obj,geometry)
                return snapshot
            options = {'target_update': update} if target_resolver is not None else {}
            outcome = method(*args, cancel_event=cancel, max_calls=call_limit - calls,
                             deadline=deadline, before_decision=before_motion,
                             after_motion=checked_fresh, event_callback=micro_event, **options)
            calls += outcome['calls']
            actions += outcome['actions']
            decision_sources.add(outcome.get('provenance', 'unverified'))
            active_target = None
            active_resolver = None
            if outcome.get('status') not in ('completed', 'actuation_completed'):
                self.last_driver_outcome=outcome
                raise MotionStopped(outcome.get('reason','microcontrol_stopped'))
        try:
            self.observer.configure_task(plan['action'])
            before = observation('approach')
            point = self._point(before, obj)
            center = [*point.xyz[:2], point.xyz[2] - 0.01]
            geometry=self.sim.grasp_geometry([center[0],center[1],point.xyz[2]-.007])
            approach_offset=geometry['approach_offset_world']
            if plan['action'] == 'pick':
                if self.held is not None:
                    raise ValueError('already_holding')
                drive(self.driver.gripper, 1)
                drive(self.driver.move_to, [center[0]+approach_offset[0], center[1]+approach_offset[1], 0.06], point.covariance_m2, target_resolver=lambda snap,p: [p.xyz[0]+approach_offset[0],p.xyz[1]+approach_offset[1],.06])
                grasp = observation('descend')
                point = self._point(grasp, obj)
                drive(self.driver.move_to, [point.xyz[0]+approach_offset[0], point.xyz[1]+approach_offset[1], point.xyz[2]-geometry['tcp_below_top_m']], point.covariance_m2, target_resolver=lambda snap,p: [p.xyz[0]+approach_offset[0],p.xyz[1]+approach_offset[1],p.xyz[2]-geometry['tcp_below_top_m']])
                grasp = observation('preclose')
                point = self._point(grasp, obj)
                self.observer.begin_phase('closing',obj)
                emit('close')
                drive(self.driver.gripper, 0)
                grasp=self.observer.begin_phase('lift', obj)
                drive(self.driver.move_to, [point.xyz[0]+approach_offset[0], point.xyz[1]+approach_offset[1], 0.06], point.covariance_m2)
                lifted = observation('verify_lift')
                feedback = self.observer.verify_lift(grasp, lifted, obj)
                if not feedback['verified']:
                    raise ValueError('grasp_not_visually_verified')
                self.held = obj
            else:
                from embodied_arm.perception.camera_perception import ObjectEstimate
                goal=ObjectEstimate.from_dict(self.observer.fixed_goal()['estimate'])
                robot = robot_only(before['robot'])
                offset = [a - b for a, b in zip(robot['ee'], center)]
                release = [goal.xyz[0] + offset[0], goal.xyz[1] + offset[1], goal.xyz[2] + 0.01 + offset[2] + 0.002]
                if not 0.008 <= release[2] <= 0.04:
                    raise ValueError('unsafe_release_geometry')
                drive(self.driver.move_to, [release[0], release[1], 0.06], goal.covariance_m2)
                observation('lower')
                drive(self.driver.move_to, release, goal.covariance_m2)
                emit('release')
                drive(self.driver.gripper, 1)
                released = observation('released')
                drive(self.driver.move_to, [release[0], release[1], 0.06], goal.covariance_m2)
                retreated = observation('verify_release')
                feedback = self.observer.verify_release(before, released, retreated, obj, list(goal.xyz))
                if not feedback['verified']:
                    raise ValueError('release_not_visually_verified')
                self.held = None
            return {'verified': True, 'action': plan['action'], 'object_id': obj, 'calls': calls, 'actions': actions, 'phases': events, 'feedback': feedback, 'source': 'rgb_only', 'controller': 'jev', 'decision_provenance': sorted(decision_sources), 'observation_retries':observation_retries}
        except MotionStopped as exc:
            return {'verified':False,'reason':str(exc),'perception_detail':getattr(self.observer,'last_error',None),'driver':getattr(self,'last_driver_outcome',None),'calls':calls,'actions':actions,'phases':events,'source':'rgb_only','decision_provenance':sorted(decision_sources),'observation_retries':observation_retries}
        finally:
            self.sim.stop()

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
    return VisualManipulationExecutor(simulation=simulation,observer=PhaseSceneObserver(source),
        driver=JevMotionDriver(simulation,controller),experimental_simulation=True)
