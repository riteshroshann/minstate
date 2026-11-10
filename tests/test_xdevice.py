import json

import torch
import yaml

from mvs import latency, xdevice
from mvs.state import payload
from mvs.trainer import Trainer


def run(tr, n):
    return [tr.step() for _ in range(n)]


def moments(tr):
    p = next(iter(tr.params.values()))
    st = tr.opt.state[p]
    return int(st['step']), st['exp_avg'], st['exp_avg_sq']


def test_w_zero_keeps_counter_and_zeroes_both_moments(cfg):
    a, b = Trainer(cfg), Trainer(cfg)
    run(a, 12)
    info = payload.migrate(a, b, 'w_zero')
    tau, m, v = moments(b)
    assert tau == 12 and not m.any() and not v.any()
    assert info['sizes']['m'] == info['sizes']['v'] == 0 and info['sizes']['theta'] == 4 * a.n_params()


def test_w_m_carries_m_exactly_and_zeroes_v(cfg):
    a, b = Trainer(cfg), Trainer(cfg)
    run(a, 12)
    payload.migrate(a, b, 'w_m')
    tau, m, v = moments(b)
    assert tau == 12 and torch.equal(m, moments(a)[1]) and not v.any()


def test_latency_round_trip_times_every_stage(cfg, tmp_path):
    a, b = Trainer(cfg), Trainer(cfg)
    run(a, 5)
    r = latency.once(a, b, 'w_vr1', str(tmp_path / 'p.mvs'))
    assert r['bytes'] > 0 and all(k in r for k in (*latency.DRAIN, *latency.RESTORE))
    rows = [{'scale': 's', 'device': 'cpu', 'strategy': 'w_vr1', 'params': 1, 'rep': i, **r, 'drain': 1.0 + i,
             'restore': 2.0} for i in range(5)]
    s = latency.summarise(rows)[0]
    assert s['reps'] == 5 and s['drain'] == 3.0 and s['drain_p5'] < s['drain'] < s['drain_p95']


def test_cross_device_pipeline_reproduces_a_direct_resume(cfg, overrides, tmp_path):
    scale = {k: yaml.safe_load(v) for k, v in (o.split('=', 1) for o in overrides)}
    e = {'base': None, 'seed': cfg.seed, 'scales': {'tiny': scale}, 'forks': [0.5], 'horizon': 20,
         'strategies': ['full', 'w_vr1'], 'conditions': {'as_trained': {}}, 'device_overrides': {},
         'snaps': str(tmp_path / 'snaps'), 'out': str(tmp_path / 'out'), 'report': str(tmp_path / 'rep'),
         'reference': 'cpu'}
    # the fixture's device is auto, which may be a GPU; the direct run has to sit on the device it is compared on
    a = Trainer(xdevice.config(e, 'tiny', 'as_trained', 'cpu'))
    t0 = cfg.max_steps // 2
    run(a, t0)
    snap = xdevice.snapshot(e, 'tiny', t0)
    snap.parent.mkdir(parents=True)
    payload.save(a, 'full', str(snap))
    xdevice.encode(e, 'cpu')
    xdevice.resume(e, 'cpu')
    got = json.loads((tmp_path / 'out' / 'cpu' / f'tiny_as_trained_{t0}_full.json').read_text())
    assert got['train'] == run(a, 20) and got['val'][str(a.t)] == a.evaluate()
    rows = xdevice.report(e)
    lossy = next(r for r in rows if r['strategy'] == 'w_vr1')
    assert 'regret' in lossy and 'dev_max' not in lossy
