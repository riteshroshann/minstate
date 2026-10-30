from pathlib import Path

import numpy as np
import pytest

from mvs.analysis import delays, regret, seed_band, steps_lost, str_steps, table

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


def test_seed_band_is_95th_percentile_of_pairwise_gaps():
    a, b, c = {10: 1.0, 20: 1.0}, {10: 1.1, 20: 1.0}, {10: 1.0, 20: 1.0, 30: 9.0}
    assert seed_band([a, b, c], 0, 100) == pytest.approx(np.percentile([0.1, 0, 0, 0, 0.1, 0], 95))
    assert seed_band([a, b], 10, 100) == 0.0


def test_str_needs_k_consecutive_steps_and_censors():
    ks = list(range(10, 310, 10))
    assert str_steps(ks, [0.0] * 30, 0.01) == 0
    late = [0.05] * 12 + [0.0] * 18
    assert str_steps(ks, late, 0.01) == 130
    lucky = [0.05] * 5 + [0.0] + [0.05] * 6 + [0.0] * 18
    assert str_steps(ks, lucky, 0.01) == 130
    assert str_steps(ks, [0.05] * 26 + [0.0] * 4, 0.01) is None
    assert str_steps(ks, [0.0] * 30, None) is None


def test_regret_integrates_only_excess():
    assert regret([10, 20, 30], [0.1, -0.5, 0.2]) == pytest.approx(10 * 0.1 + 10 * 0.2)


def test_table_reports_seed_spread_and_censoring():
    rows = []
    for seed, lam, rec in (('1', 4.0, 20), ('2', 6.0, '>290')):
        for name, mb in (('full', 12.0), ('w_vr1', 4.0)):
            rows.append({'scale': 'gpt-xs', 'params': '100', 'seed': seed, 'fork': 60, 'strategy': name, 'MB': mb,
                         'lam': 0.0 if name == 'full' else lam, 'StR': 0 if name == 'full' else rec, 'regret': 1.0,
                         'payload': mb / 12})
    rows.append({'scale': 'gpt-xs', 'params': '100', 'seed': 'band', 'fork': 60, 'strategy': 'seed_band', 'eps': 3.0,
                 'lam': 1.5})
    t = {line.split('|')[1].strip(): [c.strip() for c in line.split('|')[2:-1]] for line in table(rows).splitlines()[2:]}
    assert t['w_vr1'] == ['33%', '5.0 ± 1.0', '20 (1/2 >W)', '1.00']
    assert t['full'][2] == '0' and t['seed band (95th pct)'][1:3] == ['1.5', 'eps 3.0 mnats']
