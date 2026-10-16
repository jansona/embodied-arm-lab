"""Offline JavaScript formatter checks; no browser or model requests."""
import json
import subprocess
from pathlib import Path


def test_uncertainty_intervals_are_diagnostic_and_unresolved_is_not_success():
    source=(Path(__file__).parent/'static_so101'/'ui.js').read_text()
    helper=source[source.index('function formatAxisUncertainty'):source.index('function render(s)')]
    script=helper+"\nconsole.log(JSON.stringify([formatAxisUncertainty({axis:'x',interval_m:[-.004,.006],status:'indeterminate'}),formatAxisUncertainty({axis:'y',interval_m:[.004,.008],status:'outside'}),formatAxisUncertainty({axis:'z',interval_m:[-.002,.002],status:'inside'}),formatAxisUncertainty({axis:'x',interval_m:[NaN,0],status:'inside'})]));"
    values=json.loads(subprocess.check_output(['node','-e',script],text=True))
    assert '[-4.00, 6.00] mm' in values[0] and '精度未确定' in values[0]
    assert '容差外' in values[1] and '估计区间' in values[2]
    assert values[3]=='诊断区间不可用'
    assert '尚未确认到达' in source
    html=(Path(__file__).parent/'static_so101'/'index.html').read_text()
    assert '2σ + 1 mm' in html and '未校准为置信保证' in html and '±3 mm' in html
