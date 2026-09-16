import pytest
import throttle


def test_ninety_percent_yields_one_tenth_of_total_time(monkeypatch):
    pauses=[]
    monkeypatch.setattr(throttle.time, 'sleep', pauses.append)
    for seconds in (0.09, 0.9, 1.8):
        delay=throttle.yield_gpu(seconds, 0.9)
        assert seconds/(seconds+delay) == pytest.approx(0.9)
    assert pauses == pytest.approx([0.01, 0.1, 0.2])
    throttle.yield_gpu(1.0, 1.0)
    assert len(pauses) == 3


def test_live_control_validates_and_reloads(tmp_path):
    assert throttle.read_duty_cycle(tmp_path) == 1.0
    path=tmp_path/'gpu-duty-cycle.json'
    path.write_text('{"duty_cycle":0.9}')
    assert throttle.read_duty_cycle(tmp_path) == 0.9
    path.write_text('{"duty_cycle":0.8}')
    assert throttle.read_duty_cycle(tmp_path) == 0.8
    for invalid in ('0', '1.1', 'NaN'):
        path.write_text('{"duty_cycle":'+invalid+'}')
        with pytest.raises(ValueError):
            throttle.read_duty_cycle(tmp_path)
