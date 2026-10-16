"""Fresh RGB-only two-cube arrangement checks; never a contact-force claim."""
from embodied_arm.perception.camera_perception import EstimatedScene, PixelDetection
from embodied_arm.perception.visual_feedback import verify_release, VerificationConfig
import math

def verify_stack_release(held,released,retreated,object_id,support_id,*,now,lease_seconds=120,max_age_s=3.):
    failed={'verified':False,'reason':'support_visual_evidence_unavailable','physical_contact_verified':False,'source':'temporal_rgb_stack'}
    try:
        if type(max_age_s) not in (int,float) or not math.isfinite(max_age_s) or not 1<=max_age_s<=60:raise ValueError('invalid_observation_age')
        if support_id==object_id:raise ValueError('self_support')
        if any(s.get('source')!='rgb_only' for s in (held,released,retreated)):raise ValueError('rgb_required')
        initial=EstimatedScene.from_dict(held['scene'])
        if not 0<=now-initial.captured_at<=lease_seconds:raise ValueError('stale_support_reference')
        before=initial.require(support_id,now=initial.captured_at,max_age_s=max_age_s)
        for sample in (held,released,retreated):
            evidence_scene=EstimatedScene.from_dict(sample['scene'])
            selected=[PixelDetection.from_dict(d) for d in sample['visual_evidence']['detections'] if d.get('object_id')==support_id]
            if (len(selected)!=2 or {d.camera_id for d in selected}!=set(evidence_scene.frame_ids)
                    or any(d.frame_id!=evidence_scene.frame_ids[d.camera_id] or d.visibility!='visible' or d.confidence<.85 for d in selected)):
                raise ValueError('support_visual_evidence_unavailable')
        current=EstimatedScene.from_dict(retreated['scene'])
        support=current.require(support_id,now=now,max_age_s=max_age_s)
        carried=current.require(object_id,now=now,max_age_s=max_age_s)
        if min(before.confidence,support.confidence,carried.confidence)<.8:raise ValueError('support_uncertain')
        if max(before.position_std_m,support.position_std_m,carried.position_std_m)>.003:raise ValueError('support_uncertain')
        margin=3*(before.position_std_m+support.position_std_m)
        if math.dist(before.xyz,support.xyz)+margin>.01:raise ValueError('support_moved_or_uncertain')
        relative=[a-b for a,b in zip(carried.xyz,support.xyz)]
        relative_margin=3*(carried.position_std_m+support.position_std_m)
        if max(abs(relative[0]),abs(relative[1]))+relative_margin>.007:raise ValueError('stack_overlap_unverified')
        if abs(relative[2]-.02)+relative_margin>.006:raise ValueError('stack_height_unverified')
        verdict=verify_release(held,released,retreated,object_id,
            [support.xyz[0],support.xyz[1],support.xyz[2]+.02],target_std_m=support.position_std_m,now=now,
            config=VerificationConfig(max_age_s=max_age_s,max_semantic_age_s=lease_seconds,max_phase_s=lease_seconds))
        if not verdict.verified:return {**failed,'reason':verdict.reason}
        return {**verdict.to_dict(),'verified':True,'reason':'fresh_rgb_stack_geometry_verified',
                'support_id':support_id,'relative_top_xyz_m':relative,'relative_uncertainty_m':relative_margin,
                'physical_contact_verified':False,'source':'temporal_rgb_stack'}
    except (ValueError,KeyError,TypeError) as exc:
        allowed={'self_support','rgb_required','support_uncertain','support_moved_or_uncertain','stack_overlap_unverified','stack_height_unverified'}
        return {**failed,'reason':str(exc) if str(exc) in allowed else failed['reason']}
