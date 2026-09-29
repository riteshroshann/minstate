import json
import os
import time
from pathlib import Path

import torch
import yaml

from . import analysis, workloads
from .config import load
from .elastic import Log
from .state import payload
from .state.strategy import STUDY
from .trainer import Trainer, pick_device
from .workloads import NOISE


def spec(path):
    with open(path) as f:
        e = yaml.safe_load(f)
    e.setdefault('strategies', STUDY)
    e.setdefault('report', f"reports/{Path(e['out']).name}")
    return e


def configs(e):
    for scale, ov in e['scales'].items():
        merged = {**e.get('overrides', {}), **(ov or {})}
        yield scale, load(e['base'], [f'{k}={json.dumps(v)}' for k, v in merged.items()])


def forks(cfg, fracs):
    return [int(round(f * cfg.max_steps)) for f in fracs]


def control(cfg, d, fracs):
    if (d / 'done.json').exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    (d / 'log.jsonl').unlink(missing_ok=True)
    tr, log, snaps = Trainer(cfg), Log(str(d / 'log.jsonl')), set(forks(cfg, fracs))
    while tr.t < cfg.max_steps:
        t, t0 = tr.t, time.perf_counter()
        loss = tr.step()
        log(t=t, loss=loss, ms=(time.perf_counter() - t0) * 1e3)
        if tr.t % cfg.eval_every == 0:
            log(t=tr.t, val=tr.evaluate())
        if tr.t in snaps:
            payload.save(tr, 'full', str(d / f'snap_{tr.t}.mvs'))
    (d / 'done.json').write_text(json.dumps({'params': tr.n_params(), 'steps': tr.t}))


def source(cfg, snap, data):
    src = Trainer(cfg, data=data)
    payload.load(src, snap)
    return src


def branch(cfg, snap, name, horizon, data):
    dst = Trainer(cfg, data=data)
    info = payload.migrate(source(cfg, snap, data), dst, 'full' if name == 'noise' else name)
    if name == 'noise':
        dst.stream = NOISE
    t0, train, val = dst.t, [], {}
    for _ in range(horizon):
        train.append(dst.step())
        if dst.t % cfg.eval_every == 0:
            val[dst.t] = dst.evaluate()
    return {**info, 'strategy': name, 'codec_strategy': info['strategy'], 't0': t0, 'train': train, 'val': val}


def atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def run(path):
    e = spec(path)
    root = Path(e['out'])
    for scale, cfg in configs(e):
        sd, data = root / scale, None
        control(cfg, sd / 'control', e['forks'])
        for t0 in forks(cfg, e['forks']):
            for name in [*e['strategies'], 'noise']:
                dst = sd / 'migrations' / f'{t0}_{name}.json'
                if dst.exists():
                    continue
                data = data or workloads.build(cfg, pick_device(cfg, 0))
                r = branch(cfg, str(sd / 'control' / f'snap_{t0}.mvs'), name, e['horizon'], data)
                atomic(dst, r)
                print(f"{scale} fork {t0} {name:12s} {r['file_bytes'] / 1e6:9.2f} MB  final loss {r['train'][-1]:.4f}")
    rows = analysis.report(root, Path(e['report']))
    print(analysis.table(rows))
    return rows


def first_update(tr):
    c, st = tr.cfg, tr.opt.state
    tau = int(st[next(iter(tr.params.values()))]['step'])
    tr.grads(tr.t, tr.stream)
    u, m0, v0 = [], [], []
    for p in tr.params.values():
        m, v, g = st[p]['exp_avg'], st[p]['exp_avg_sq'], p.grad
        m1, v1 = c.beta1 * m + (1 - c.beta1) * g, c.beta2 * v + (1 - c.beta2) * g * g
        mh, vh = m1 / (1 - c.beta1 ** (tau + 1)), v1 / (1 - c.beta2 ** (tau + 1))
        u.append((mh / (vh.sqrt() + c.eps)).flatten())
        m0.append((m / (1 - c.beta1 ** tau) if tau else torch.zeros_like(m)).flatten())
        v0.append((v / (1 - c.beta2 ** tau) if tau else torch.zeros_like(v)).flatten())
    tr.opt.zero_grad(set_to_none=True)
    return torch.cat(u), torch.cat(m0), torch.cat(v0), tr.lr_at(tr.t)


def fidelity(cfg, snap, name, data):
    ref = source(cfg, snap, data)
    dst = Trainer(cfg, data=data)
    payload.migrate(source(cfg, snap, data), dst, name)
    u0, m0, v0, lr0 = first_update(ref)
    u1, m1, v1, lr1 = first_update(dst)
    root0, root1 = v0.sqrt(), v1.sqrt()
    pos = root0 > 0
    return {'strategy': name, 'cos': torch.nn.functional.cosine_similarity(u1, u0, 0).item(),
            'step_ratio': (lr1 * u1.norm() / (lr0 * u0.norm())).item(), 'm_ratio': (m1.norm() / m0.norm()).item(),
            'sqrt_v_err': ((root1[pos] - root0[pos]).abs() / root0[pos]).median().item()}


def diagnose(path):
    e, rows = spec(path), []
    root = Path(e['out'])
    for scale, cfg in configs(e):
        data = workloads.build(cfg, pick_device(cfg, 0))
        for t0 in forks(cfg, e['forks']):
            snap = str(root / scale / 'control' / f'snap_{t0}.mvs')
            for name in e['strategies']:
                rows.append({'scale': scale, 'fork': t0, **fidelity(cfg, snap, name, data)})
                print(' '.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in rows[-1].values()))
    Path(e['report']).mkdir(parents=True, exist_ok=True)
    analysis.write_csv(rows, Path(e['report']) / 'diagnostics.csv')
    return rows
