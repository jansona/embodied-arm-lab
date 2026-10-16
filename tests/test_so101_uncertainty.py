import pytest
from so101_uncertainty import axis_bands

def covariance(sigma):return [[sigma*sigma if i==j else 0 for j in range(3)] for i in range(3)]
def test_near_boundary_does_not_block_large_axis_or_count_as_reached():
    result=axis_bands([-.0031,-.05,-.005],covariance(.00233))
    assert [x['status'] for x in result]==['indeterminate','outside','indeterminate']
    assert next(i for i,x in enumerate(result) if x['status']=='outside')==1

def test_zero_point_error_with_large_uncertainty_is_not_success():
    assert all(x['status']=='indeterminate' for x in axis_bands([0,0,0],covariance(.00233)))
def test_small_uncertainty_can_verify_original_3mm_band():
    assert all(x['status']=='inside' for x in axis_bands([0,0,0],covariance(.0004)))
@pytest.mark.parametrize('sign',[-1,1])
def test_both_signs_same_axis_status(sign):
    assert axis_bands([sign*.02,0,0],covariance(.002))[0]['status']=='outside'
def test_invalid_variance_rejected():
    with pytest.raises(ValueError):axis_bands([0,0,0],[[-1,0,0],[0,1,0],[0,0,1]])
@pytest.mark.parametrize('kwargs',[{'servo_allowance':-1},{'servo_allowance':float('nan')},{'tolerance':0}])
def test_invalid_allowance_rejected(kwargs):
    with pytest.raises(ValueError):axis_bands([0,0,0],covariance(.002),**kwargs)
