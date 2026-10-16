"""Image-coordinate-only crop geometry; no object poses or simulator reads."""
import math

def bounds_from_detections(images,detections,required_objects):
    result={}
    for image in images:
        cid=image['camera_id'];width=image['width'];height=image['height']
        points=[]
        for name in required_objects:
            matches=[d for d in detections if d.camera_id==cid and d.object_id==name]
            if len(matches)!=1 or matches[0].uv is None or matches[0].visibility!='visible':raise ValueError('Required crop landmark missing')
            points.append(matches[0].uv)
        radius=max(24,math.ceil(width*.07))
        result[cid]=[max(0,math.floor((min(p[0] for p in points)-radius)/8)*8),
                     max(0,math.floor((min(p[1] for p in points)-radius)/8)*8),
                     min(width,math.ceil((max(p[0] for p in points)+radius+1)/8)*8),
                     min(height,math.ceil((max(p[1] for p in points)+radius+1)/8)*8)]
    return result

def frustum(bounds,width,height,near=.01,far=2.,fov=50):
    if len(bounds)!=4 or any(type(v) is not int for v in bounds):raise ValueError('Integer crop bounds required')
    x0,y0,x1,y1=bounds
    if not (0<=x0<x1<=width and 0<=y0<y1<=height):raise ValueError('Crop outside native sensor')
    top=near*math.tan(math.radians(fov)/2);right=top*width/height
    return (-right+2*right*x0/width,-right+2*right*x1/width,
            top-2*top*y1/height,top-2*top*y0/height,near,far)
