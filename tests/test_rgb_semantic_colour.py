import numpy as np
import pytest
from embodied_arm.perception.rgb_semantic_colour import semantic_colour_mask


@pytest.mark.parametrize('name,rgb', [('red',[210,24,24]),('blue',[24,70,225]),('goal',[18,195,55]),
                                     ('red',[176,23,25]),('red',[110,12,12]),('blue',[16,43,140])])
def test_expected_saturated_colours_and_shadowed_faces(name,rgb):
    assert semantic_colour_mask(np.array([[rgb]]),name)[0,0]


@pytest.mark.parametrize('rgb',[[156,123,15],[220,175,0],[180,180,180],[0,0,0],[250,130,0],
                                [20,220,20],[20,30,220]])
def test_yellow_jaw_gray_or_other_hues_are_not_red(rgb):
    assert not semantic_colour_mask(np.array([[rgb]]),'red')[0,0]


@pytest.mark.parametrize('name', ['red','blue','goal'])
def test_semantic_hue_gate_is_strict_subset_of_existing_mask(name):
    rng=np.random.default_rng(374)
    rgb=rng.integers(0,256,size=(100,100,3)).astype(float)
    index={'red':0,'blue':2,'goal':1}[name]
    d=rgb[:,:,index];a,b=[rgb[:,:,i] for i in range(3) if i!=index]
    old=(d>=55)&(d-(a+b)/2>=32)&(d>1.25*a)&(d>1.25*b)
    assert not np.any(semantic_colour_mask(rgb,name)&~old)
