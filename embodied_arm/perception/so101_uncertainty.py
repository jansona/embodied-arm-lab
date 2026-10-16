"""Conservative scheduling bands, not calibrated confidence or reached proof.

Image covariance is an approximate measurement model and omits systematic error.
Two sigma plus 1 mm FK/servo allowance is a diagnostic envelope, not a guarantee.
No action direction is selected here; Jev retains positive/negative/hold choices.
"""
import math

def axis_bands(error,covariance,tolerance=.003,servo_allowance=.001):
    if (len(error)!=3 or len(covariance)!=3 or any(len(row)!=3 for row in covariance)
            or any(not math.isfinite(v) for row in covariance for v in row)
            or any(abs(covariance[i][j]-covariance[j][i])>1e-12 for i in range(3) for j in range(3))
            or not math.isfinite(tolerance) or tolerance<=0 or not math.isfinite(servo_allowance) or servo_allowance<0):raise ValueError('Invalid uncertainty envelope')
    result=[]
    for i,e in enumerate(error):
        variance=covariance[i][i]
        if not math.isfinite(e) or not math.isfinite(variance) or variance<0:raise ValueError('Invalid measurement')
        margin=2*math.sqrt(variance)+servo_allowance
        lower,upper=e-margin,e+margin
        status='outside' if lower>tolerance or upper < -tolerance else 'inside' if lower>=-tolerance and upper<=tolerance else 'indeterminate'
        result.append({'axis':'xyz'[i],'error_m':e,'margin_m':margin,'interval_m':[lower,upper],'status':status})
    return result
