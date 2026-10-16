"""Independent offline evaluation only; never imported by model/control code."""
import json,math
from pathlib import Path
import numpy as np
from embodied_arm.perception.camera_perception import CameraCalibration,_closest_point
from embodied_arm.simulation.so101_simulation import SO101Simulation,camera_calibration

def evaluate(diagnostic):
    calibrations=[CameraCalibration.from_dict(camera_calibration(k)) for k in ('left','right')]
    simulation=SO101Simulation(seed=0)
    try:oracle=simulation.evaluation_state()
    finally:simulation.close()
    report={'purpose':'independent_evaluation_only','not_for_model_input':True,
            'oracle_source':'separate_deterministic_seed0_initial_scene_replica','objects':{}}
    for name in ('red','blue','goal'):
        selected=[next(d for d in diagnostic['detections'] if d['object_id']==name and d['camera_id']==k) for k in ('left','right')]
        pixels=np.array([v*s for d in selected for v,s in zip(d['uv_1000'],(.639,.479))])
        point,separation,angle=_closest_point(calibrations,pixels)
        J=np.empty((3,4))
        for column in range(4):
            delta=np.zeros(4);delta[column]=.01
            J[:,column]=(_closest_point(calibrations,pixels+delta)[0]-_closest_point(calibrations,pixels-delta)[0])/.02
        sigmas=np.array([max(2,d['sigma_1000']*.639) for d in selected for _ in range(2)])
        covariance=(J*sigmas)@(J*sigmas).T+np.eye(3)*(separation/2)**2
        floorcov=(J*2)@(J*2).T
        truth=np.array(oracle['ground_truth_objects'][name] if name!='goal' else oracle['ground_truth_goal'])
        truth[2]+=.01 if name!='goal' else .0005
        reprojection=max(math.dist(c.project(tuple(float(x) for x in point)),pixels[i*2:i*2+2]) for i,c in enumerate(calibrations))
        truth_pixels=[list(c.project(tuple(float(x) for x in truth))) for c in calibrations]
        report['objects'][name]={'estimated_landmark_xyz':point.tolist(),'independent_true_landmark_xyz':truth.tolist(),
            'error_mm':1000*float(np.linalg.norm(point-truth)),'error_xyz_mm':((point-truth)*1000).tolist(),
            'position_std_mm':1000*float(np.sqrt(np.linalg.eigvalsh(covariance).max())),
            'two_pixel_floor_std_mm':1000*float(np.sqrt(np.linalg.eigvalsh(floorcov).max())),
            'reprojection_error_px':reprojection,'ray_separation_mm':separation*1000,'ray_angle_deg':angle,
            'reported_native_pixel_sigmas':sigmas[::2].tolist(),'native_pixels':pixels.reshape(2,2).tolist(),
            'independent_true_pixels':truth_pixels,'jacobian_m_per_pixel':J.tolist()}
    return report
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('diagnostic');p.add_argument('output');a=p.parse_args()
    report=evaluate(json.loads(Path(a.diagnostic).read_text()));Path(a.output).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:{f:v for f,v in d.items() if f in ('error_mm','position_std_mm','two_pixel_floor_std_mm','reprojection_error_px','ray_separation_mm')} for k,d in report['objects'].items()}))
