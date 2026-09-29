from pathlib import Path

import numpy as np
import pytest

from mvs.analysis import delays, noise_floor, read_csv, recovered, steps_lost, table

T = np.arange(1000)
CTRL = 1.0 + np.exp(-T / 250.0)


def test_identical_run_loses_exactly_zero():
    d = delays(CTRL, CTRL[400:700], 400)
    assert (d == 0).all() and steps_lost(d)[0] == 0.0


def test_lagging_run_is_measured_in_steps():
    lag = 1.0 + np.exp(-(T - 5) / 250.0)
    lam, tail = steps_lost(delays(CTRL, lag[400:700], 400))
    assert tail == pytest.approx(5.0, abs=0.3) and lam == pytest.approx(tail)


def test_leading_run_costs_only_its_passes():
    ahead = 1.0 + np.exp(-(T + 8) / 250.0)
    lam, tail = steps_lost(delays(CTRL, ahead[400:700], 400), passes=20)
    assert tail < -6 and lam == 20


def test_noise_floor_and_recovery():
    assert noise_floor([0.0005, -0.001]) == 2e-3 and noise_floor([0.01, -0.03]) == 0.03
    ks = [10, 20, 30, 40]
    assert recovered(ks, [0.01, 0.001, 0.0, 0.001], 0.002) == 20
    assert recovered(ks, [0.001, 0.0, 0.0, 0.01], 0.002) == 'never'
    assert recovered(ks, [0.0] * 4, 0.002) == 0


def test_paper_table_is_reproduced_from_appendix_b():
    rows = read_csv(str(Path(__file__).resolve().parents[1] / 'reports' / 'paper' / 'summary.csv'))
    assert len(rows) == 156
    t = {line.split('|')[1].strip(): [c.strip() for c in line.split('|')[2:-1]] for line in table(rows).splitlines()[2:]}
    assert t['w_vr1'] == ['34%', '11.8', '7.1', '6.6', '8.5', '8.9']
    assert t['w_warm'][3] == '61.2' and t['w_mq8_vlog8'][:2] == ['50%', '0.0'] and t['wbf16_vr1'][0] == '17%'
