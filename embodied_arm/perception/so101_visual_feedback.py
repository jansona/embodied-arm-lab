"""Temporal RGB evidence for manipulation; never reads simulator oracle state."""
from __future__ import annotations
import math
from embodied_arm.perception.camera_perception import EstimatedScene
from embodied_arm.control.so101_control import robot_only, MAX_POSITION_STD_M


def verify_visual_lift(before,after,object_id,*,now,max_age_s=30.):
    """Conservative evidence, not a claim of force sensing or certain grasp.

    Both independently observed landmarks must rise with encoder-derived TCP.
    Missing/occluded objects fail closed. The initial frame may be older than the
    current freshness window, but observation order and the latest frame matter.
    """
    if object_id not in ('red','blue'):raise ValueError('Unknown object')
    for snapshot in (before,after):
        if snapshot.get('schema')!='so101-camera-observation-v1' or snapshot.get('source')!='rgb_only':
            raise ValueError('RGB observation required')
    first=EstimatedScene.from_dict(before['scene']);last=EstimatedScene.from_dict(after['scene'])
    if last.captured_at<=first.captured_at:raise ValueError('Independent later RGB frame required')
    a=first.require(object_id,now=first.captured_at,max_age_s=max_age_s)
    b=last.require(object_id,now=now,max_age_s=max_age_s)
    if max(a.position_std_m,b.position_std_m)>MAX_POSITION_STD_M:raise ValueError('Uncertain visual position')
    r0=robot_only(before['robot']);r1=robot_only(after['robot'])
    if r0['epoch']!=r1['epoch']:raise ValueError('Reset or stop invalidates lift evidence')
    delta=[y-x for x,y in zip(a.xyz,b.xyz)];tcp=[y-x for x,y in zip(r0['ee'],r1['ee'])]
    mismatch=math.dist(delta,tcp)
    # Explicit uncertainty-aware margins. This never supplies an action choice.
    margin=3*math.sqrt(a.position_std_m**2+b.position_std_m**2)
    lifted=delta[2]>.015+margin and tcp[2]>.015
    follows=mismatch<.008+margin
    nearby=math.dist(b.xyz,r1['ee'])<.035+margin
    return {'source':'temporal_rgb','verified':bool(lifted and follows and nearby),
            'object_id':object_id,'visual_rise_m':delta[2],'encoder_rise_m':tcp[2],
            'motion_mismatch_m':mismatch,'uncertainty_margin_m':margin,
            'reason':'visual_lift_consistent' if lifted and follows and nearby else 'visual_lift_not_verified',
            'physical_contact_verified':False}
