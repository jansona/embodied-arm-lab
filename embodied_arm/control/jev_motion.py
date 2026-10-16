"""Bounded Jev-selected servo motions; no scene/oracle reads or physics stepping.

The caller owns RGB target provenance, freshness, obstacle/path clearance, and
visual grasp verification. Fixed XYZ scheduling selects an axis, never direction.
"""
import math
import threading
import time
import numpy as np
from embodied_arm.control.so101_control import robot_only
from embodied_arm.providers.so101_jev import build_observation, validate_decision
from embodied_arm.perception.so101_uncertainty import axis_bands


class JevMotionDriver:
    # Contract and real-servo fixture tests validate this driver, not live model capability.
    validated = True
    def __init__(self, sim, controller, *, clock=time.monotonic, sleep=time.sleep):
        self.sim, self.controller = sim, controller
        self.clock, self.sleep = clock, sleep
        self._lock = threading.Lock()

    def move_to(self, target_xyz, covariance_m2, *, cancel_event=None,
                max_calls=48, deadline=None, event_callback=None,
                before_decision=None, after_motion=None):
        return self._run('cartesian', target_xyz, covariance_m2, cancel_event,
                         max_calls, deadline, event_callback, before_decision, after_motion)

    def gripper(self, target_openness, *, cancel_event=None, max_calls=1,
                deadline=None, event_callback=None, before_decision=None, after_motion=None):
        return self._run('gripper', target_openness, None, cancel_event,
                         max_calls, deadline, event_callback, before_decision, after_motion)

    def _check(self, cancel, deadline):
        if cancel is not None and cancel.is_set():
            raise ValueError('cancelled')
        if self.clock() >= deadline:
            raise ValueError('deadline')

    def _robot(self):
        robot = robot_only(self.sim.proprioception())
        if robot['busy'] or not robot['paused']:
            raise ValueError('robot_not_idle_paused')
        return robot

    def _unchanged(self, before):
        after = self._robot()
        if (any(after[k] != before[k] for k in ('epoch', 'actions'))
                or math.dist(after['ee'], before['ee']) > .001
                or max(abs(a-b) for a,b in zip(after['joints'], before['joints'])) > .01
                or abs(after['gripper_openness']-before['gripper_openness']) > .01):
            raise ValueError('robot_changed_during_decision')

    def _actuate(self, action, values, cancel, deadline, result):
        self._check(cancel, deadline)
        before = self.sim.proprioception()
        self.sim.resume()
        source=getattr(self.controller,'provenance','unverified')
        origin='offline_fixture' if any(tag in source.lower() for tag in ('fake','fixture','mock','offline')) else 'jev'
        self.sim.manual(action, values, origin=origin)
        result['actions'] += 1
        motion_deadline = min(deadline, self.clock()+5.)
        while self.sim.proprioception()['busy']:
            self._check(cancel, deadline)
            if self.clock() >= motion_deadline:
                raise ValueError('motion_timeout')
            self.sleep(.02)  # The app, never this driver, advances physics.
        self._check(cancel, deadline)
        self.sim.pause_idle()
        after = self.sim.proprioception()
        if after['epoch'] != before['epoch'] or after['actions'] != before['actions']+1:
            raise ValueError('robot_changed_during_motion')

    @staticmethod
    def _perception_callback(callback):
        try:
            if callback() is False:
                raise ValueError('Rejected observation')
        except Exception:
            raise ValueError('perception_rejected') from None

    def _run(self, kind, target, covariance, cancel, max_calls, deadline, callback, before_decision, after_motion):
        result = {'status':'stopped', 'reason':'unavailable', 'calls':0, 'actions':0,
                  'controller':'jev', 'provenance':getattr(self.controller, 'provenance', 'unverified'),
                  'source':'caller_rgb_target', 'grasp_verified':False,
                  'axis_schedule':'fixed_xyz_clearly_unfinished_axes',
                  'amplitude_schedule':'up_to_8mm_avoid_indeterminate_landing_no_direction_selection'}
        if not self._lock.acquire(blocking=False):
            return {**result, 'reason':'driver_busy'}
        try:
            if type(max_calls) is not int or not 1 <= max_calls <= 256:
                raise ValueError('invalid_budget')
            deadline = self.clock()+90 if deadline is None else deadline
            if type(deadline) not in (int,float) or not math.isfinite(deadline):
                raise ValueError('invalid_budget')
            if kind == 'cartesian':
                if (not isinstance(target,(list,tuple)) or len(target)!=3
                        or any(type(x) not in (int,float) or not math.isfinite(x) for x in target)):
                    raise ValueError('invalid_target')
                target = list(target)
                covariance = np.asarray(covariance, dtype=float)
                if (covariance.shape != (3,3) or not np.isfinite(covariance).all()
                        or not np.allclose(covariance,covariance.T,atol=1e-12,rtol=0)
                        or np.linalg.eigvalsh(covariance).min() < -1e-15):
                    raise ValueError('invalid_covariance')
                covariance = covariance.tolist()
            elif type(target) not in (int,float) or target not in (0,1):
                raise ValueError('invalid_target')
            for _ in range(max_calls+1):
                self._check(cancel, deadline)
                robot = self._robot()
                if kind == 'cartesian':
                    if any(not lo<=x<=hi for x,(lo,hi) in zip(target,robot['workspace'])):
                        raise ValueError('target_outside_workspace')
                    bands = axis_bands([a-b for a,b in zip(target,robot['ee'])],covariance)
                    result['axis_uncertainty'] = bands
                    if any(b['margin_m'] >= .003 for b in bands):
                        raise ValueError('precision_floor')
                    if all(b['status']=='inside' for b in bands):
                        result.update(status='completed',reason='estimated_uncertainty_envelope_within_tolerance')
                        return result
                    if not any(b['status']=='outside' for b in bands):
                        raise ValueError('uncertainty_unresolved')
                    axis = next(i for i,b in enumerate(bands) if b['status']=='outside')
                    local = {'source':'rgb_only','kind':kind,'target_xyz':target,'tolerance_m':.003,'step_m':.008}
                else:
                    axis = None
                    local = {'source':'rgb_only','kind':kind,'target_openness':target,'tolerance':.02}
                if result['calls'] >= max_calls:
                    raise ValueError('call_budget_exhausted')
                if before_decision:
                    self._perception_callback(before_decision)
                self._check(cancel,deadline)
                self._unchanged(robot)
                observation = build_observation(local,robot,axis)
                result['calls'] += 1
                raw = self.controller.decide(observation)
                decision = validate_decision(raw.to_dict() if hasattr(raw,'to_dict') else raw, observation)
                self._check(cancel,deadline)
                if callback:
                    callback({'phase':'jev_decision','kind':kind,'axis_index':axis,
                              'accepted':decision.accepted,'action':decision.action,
                              'confidence':decision.confidence,'provenance':result['provenance']})
                self._check(cancel,deadline)
                self._unchanged(robot)
                if not decision.accepted:
                    raise ValueError('controller_rejected')
                if decision.action == 'hold':
                    if kind == 'gripper':
                        result.update(status='completed',reason='encoder_within_tolerance')
                        return result
                    continue
                if kind == 'cartesian':
                    sign = 1 if decision.action.endswith('_plus') else -1
                    xyz = list(robot['ee'])
                    remaining = abs(target[axis]-xyz[axis])
                    margin = bands[axis]['margin_m']
                    step = min(.008,remaining)
                    residual = remaining-step
                    # Avoid landing in the unchanged uncertainty band. This
                    # chooses amplitude only after Jev chose the direction.
                    if .003-margin < residual <= .003+margin:
                        step = remaining-(.003+margin+.001)
                    if not 0 < step <= .008000001:
                        raise ValueError('unsafe_step')
                    xyz[axis] += sign*step
                    if math.dist(xyz,robot['ee']) > .008000001:
                        raise ValueError('unsafe_step')
                    self._actuate('xyz',xyz,cancel,deadline,result)
                else:
                    self._actuate(decision.action,None,cancel,deadline,result)
                if after_motion:
                    self._perception_callback(after_motion)
                if callback:
                    callback({'phase':'motion_completed','kind':kind,'actions':result['actions']})
                self._check(cancel,deadline)
                if kind == 'gripper':
                    # A physical object can block closure before encoder zero.
                    result.update(status='completed',reason='actuation_completed')
                    return result
            raise ValueError('call_budget_exhausted')
        except Exception as exc:
            allowed = {'cancelled','deadline','motion_timeout','robot_not_idle_paused',
                       'robot_changed_during_decision','robot_changed_during_motion','invalid_budget','invalid_target',
                       'invalid_covariance','target_outside_workspace','uncertainty_unresolved',
                       'call_budget_exhausted','controller_rejected','unsafe_step','perception_rejected','precision_floor'}
            result['reason'] = str(exc) if str(exc) in allowed else 'motion_or_controller_error'
            self.sim.stop()
            return result
        finally:
            self._lock.release()
