"""Opt-in scale-profile tests use only rendered 2D RGB geometry, never physics."""
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pytest

import rgb_face_refinement as baseline
import rgb_scaled_refinement as scaled

TOP = [(300,210),(316,200),(333,212),(316,223)]


def image(scale=1,label='red',case='clean'):
    width,height=640*scale,480*scale; aa=4
    im=Image.new('RGB',(width*aa,height*aa),(210,220,230));draw=ImageDraw.Draw(im)
    background=(210,220,230)
    bright,dark,medium=(((210,24,24),(135,15,16),(160,20,20)) if label=='red' else
                        ((24,70,225),(16,43,140),(20,53,165)))
    if case=='ambiguous_brightness':
        bright=tuple(v+4 for v in dark);medium=dark
    def poly(points,colour):
        draw.polygon([(round(x*scale*aa),round(y*scale*aa)) for x,y in points],fill=colour)
    def cube(offset=0):
        p0,p1,p2,p3=[(x+offset,y) for x,y in TOP]
        poly([p0,p3,(p3[0],p3[1]+18),(p0[0],p0[1]+18)],dark)
        poly([p3,p2,(p2[0],p2[1]+18),(p3[0],p3[1]+18)],medium)
        poly([p0,p1,p2,p3],bright)
    cube()
    if case=='second_object':cube(44)
    if case=='occlusion':poly([(311,205),(322,205),(322,217),(311,217)],background)
    if case=='missing_corner':poly([(295,203),(309,203),(307,216),(295,216)],background)
    if case=='split_face':poly([(313,196),(319,196),(319,225),(313,225)],background)
    im=im.resize((width,height),Image.Resampling.LANCZOS)
    out=BytesIO();im.save(out,'JPEG',quality=94)
    return out.getvalue()


def refine(scale=1,label='red',case='clean',coarse=(316,223),module=scaled):
    return module.refine_rgb_landmark(image(scale,label,case),object_id=label,
                                     uv_1000=[coarse[0]*scale/(640*scale-1)*1000,
                                              coarse[1]*scale/(480*scale-1)*1000],
                                     confidence=.98,width=640*scale,height=480*scale)


def centre():
    p=np.asarray(TOP,dtype=float);a,b=p[2]-p[0],p[3]-p[1]
    t=np.linalg.solve(np.column_stack((a,-b)),p[1]-p[0])[0]
    return p[0]+t*a


@pytest.mark.parametrize('scale',[1,2,3])
@pytest.mark.parametrize('label',['red','blue'])
def test_clean_faces_are_consistent_in_native_coordinates_across_resolutions(scale,label):
    result=refine(scale,label)
    assert result.accepted,result.reason
    np.testing.assert_allclose(np.asarray(result.uv)/scale,centre(),atol=.9)
    assert result.source=='hybrid_qwen_region_rgb_geometry_scaled'
    assert result.profile=='footprint_scaled_v1'
    assert result.footprint_scale>=1
    assert result.normalized_edge_rms<=1
    assert result.max_corner_gap_px<=2*result.footprint_scale
    assert result.pixel_sigma_px>=max(1,result.edge_rms_px*2+.5)
    assert result.pixel_sigma_px>=result.max_edge_rms_px*2+.5
    assert all(len(c)==2 for c in result.corners)
    json.dumps(result.to_dict(),allow_nan=False)


@pytest.mark.parametrize('scale',[1,2,3])
@pytest.mark.parametrize('case',['occlusion','missing_corner','split_face','ambiguous_brightness'])
def test_occlusion_corner_loss_and_photometric_ambiguity_fail_at_every_resolution(scale,case):
    result=refine(scale,case=case)
    assert not result.accepted,(scale,case,result.to_dict())
    assert result.uv is None and result.pixel_sigma_px is None


@pytest.mark.parametrize('scale',[1,2,3])
def test_same_colour_second_object_and_clipped_ai_roi_fail(scale):
    ambiguous=refine(scale,case='second_object',coarse=(334,223))
    assert not ambiguous.accepted
    assert ambiguous.reason in ('ambiguous_coloured_regions','roi_clipped')
    clipped=refine(scale,coarse=(357,223))
    assert not clipped.accepted
    assert clipped.reason in ('roi_clipped','no_coloured_region_near_ai_point')


def rectangle(x,y,width,height):
    ys,xs=np.indices((height,width))
    return np.column_stack((xs.ravel()+x,ys.ravel()+y)).astype(float)


def test_bounded_tiny_patches_record_area_distance_and_do_not_change_main_pixels():
    main=rectangle(10,10,50,40)
    islands=[rectangle(14,53,3,5),rectangle(17,65,2,5),rectangle(20,70,1,1)]
    selected,areas,fraction,distance=scaled._select_face([main,*islands])
    assert np.array_equal(selected,main)
    assert areas==(15,10,1)
    assert fraction==26/2000
    assert 0<distance<=math.sqrt(len(main))


@pytest.mark.parametrize('islands',[
    [rectangle(10,60,20,20)],  # A competing full face
    [rectangle(10+8*n,54,4,5) for n in range(4)], # Individually tiny, collectively4%
    [rectangle(10+2*n,53,1,1) for n in range(9)], # Count includes single pixels
    [rectangle(150,100,2,2)], # Small but spatially remote
    [rectangle(10,55,35,1)], # Small area but long, structured bright edge
])
def test_genuine_second_face_aggregate_patches_and_remote_structure_rejected(islands):
    with pytest.raises(ValueError,match='ambiguous_face_patches'):
        scaled._select_face([rectangle(10,10,50,40),*islands])


def test_single_pixel_components_are_counted_instead_of_silently_dropped():
    pixels=np.zeros((30,30),dtype=bool);pixels[2:12,2:12]=True
    for x in (15,18,21,24):pixels[20,x]=True
    assert sorted(map(len,scaled._all_components(pixels)))==[1,1,1,1,100]


def test_geometric_noise_threshold_scales_with_face_footprint_not_declared_resolution():
    # Same observed polygon, no image dimensions passed into the geometry fit.
    contour=[]
    quad=np.array([[0.,0.],[40.,0.],[40.,30.],[0.,30.]])
    for edge,(a,b) in enumerate(zip(quad,np.roll(quad,-1,axis=0))):
        delta=b-a;n=np.array([-delta[1],delta[0]])/np.linalg.norm(delta)
        for t in np.linspace(0,1,70):
            contour.append(a+t*delta+n*.8*math.sin(t*math.pi*8))
    contour=np.asarray(contour)
    first=scaled._quadrilateral(contour)
    larger=scaled._quadrilateral(contour*3)
    assert larger[3]==pytest.approx(first[3]*3,rel=.01)
    assert larger[1]==pytest.approx(first[1]*3,rel=.01)
    assert larger[1]/larger[3]==pytest.approx(first[1]/first[3],rel=.01)
    with pytest.raises(ValueError,match='irregular_face_boundary'):
        baseline._quadrilateral(contour*3)


def test_baseline_module_and_default_source_are_unchanged():
    result=refine(module=baseline)
    assert result.accepted
    assert result.source=='hybrid_qwen_region_rgb_geometry'
    assert not hasattr(result,'footprint_scale')
    source=Path(baseline.__file__).read_text()
    assert 'rgb_scaled_refinement' not in source
