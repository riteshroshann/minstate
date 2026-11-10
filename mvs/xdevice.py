import json
import statistics
import time
from pathlib import Path

import numpy as np
import yaml

from . import analysis, manifest
from .config import load
from .experiment import atomic, forks
from .state import payload
from .trainer import Trainer


def spec(path):
    with open(path) as f:
        e = yaml.safe_load(f)
    e.setdefault('seed', 1337)
    e.setdefault('conditions', {'as_trained': {}})
    e.setdefault('device_overrides', {})
    return e


def config(e, scale, cond, device):
    merged = {**(e['scales'][scale] or {}), **(e['conditions'][cond] or {}), **e['device_overrides'].get(device, {}),
              'seed': e['seed'], 'device': device}
    return load(e['base'], [f'{k}={json.dumps(v)}' for k, v in merged.items()])


def cells(e):
    for scale in e['scales']:
        for t0 in forks(config(e, scale, next(iter(e['conditions'])), 'cpu'), e['forks']):
            for name in e['strategies']:
                yield scale, t0, name


def snapshot(e, scale, t0):
    return Path(e['snaps']) / scale / f"s{e['seed']}" / 'control' / f'snap_{t0}.mvs'


def wire_path(e, scale, t0, name):
    return Path(e['out']) / 'payloads' / f'{scale}_{t0}_{name}.mvs'


def encode(e, device):
    """Source side: write one payload per (scale, fork, strategy) from the control's full snapshot."""
    cond = next(iter(e['conditions']))
    for scale, t0, name in cells(e):
        dst = wire_path(e, scale, t0, name)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tr = Trainer(config(e, scale, cond, device))
        payload.load(tr, str(snapshot(e, scale, t0)))
        info = payload.save(tr, name, str(dst))
        print(f'encoded {dst.name} on {tr.device}  {info["file_bytes"] / 1e6:.2f} MB', flush=True)


def resume(e, device):
    """Destination side: load each payload on this device and train on, recording held-out loss."""
    for cond in e['conditions']:
        for scale, t0, name in cells(e):
            out = Path(e['out']) / device / f'{scale}_{cond}_{t0}_{name}.json'
            if out.exists():
                continue
            cfg = config(e, scale, cond, device)
            tr = Trainer(cfg)
            info = payload.load(tr, str(wire_path(e, scale, t0, name)))
            # the weights are identical on arrival, so a gap here is the device's evaluation arithmetic alone
            val, train, ms = {tr.t: tr.evaluate()}, [], []
            for _ in range(e['horizon']):
                c = time.perf_counter()
                train.append(tr.step())
                ms.append((time.perf_counter() - c) * 1e3)
                if tr.t % cfg.eval_every == 0:
                    val[tr.t] = tr.evaluate()
            atomic(out, {'scale': scale, 'cond': cond, 't0': t0, 'strategy': name, 'device': device, 'val': val,
                         'train': train, 'step_ms_median': statistics.median(ms), 'load_ms': info['ms'],
                         'file_bytes': info['file_bytes'], 'passes': info['passes'],
                         'manifest': {**manifest.environment(tr.device), 'seed': cfg.seed, 'T': t0, 'arm': name}})
            print(f'{device} {scale} {cond} fork {t0} {name:12s} val {val[max(val)]:.4f}', flush=True)


def load_runs(e):
    runs = {}
    for p in Path(e['out']).glob('*/*.json'):
        r = json.loads(p.read_text())
        runs[(r['scale'], r['cond'], r['t0'], r['strategy'], r['device'])] = r
    return runs


def gaps(a, b):
    ts = sorted(set(a['val']) & set(b['val']), key=int)
    return [int(t) for t in ts], np.array([a['val'][t] - b['val'][t] for t in ts])


def bands(path):
    """Seed-band eps in nats per (scale, fork), from the main sweep's summary."""
    if not path or not Path(path).exists():
        return {}
    return {(r['scale'], int(r['fork'])): float(r['eps']) / 1e3 for r in analysis.read_csv(path)
            if r['strategy'] == 'seed_band' and r['eps'] != ''}


def report(e):
    runs, eps, rows = load_runs(e), bands(e.get('band')), []
    ref = e['reference']
    for (scale, cond, t0, name, device), r in sorted(runs.items()):
        row = {'scale': scale, 'cond': cond, 'fork': t0, 'strategy': name, 'device': device,
               'device_name': r['manifest']['device_name'], 'torch': r['manifest']['torch'],
               'step_ms': round(r['step_ms_median'], 1), 'eps': eps.get((scale, t0), '')}
        # same payload continued on the reference device: what the device change alone does
        same = runs.get((scale, cond, t0, name, ref))
        if same and device != ref:
            ts, g = gaps(r, same)
            row.update(arrival=round(1e3 * abs(g[0]), 4), dev_max=round(1e3 * np.abs(g[1:]).max(), 3),
                       dev_mean=round(1e3 * np.abs(g[1:]).mean(), 3), dev_end=round(1e3 * g[-1], 3))
        # full state on the same device: what the strategy costs on this device
        full = runs.get((scale, cond, t0, 'full', device))
        if full and name != 'full':
            ts, g = gaps(r, full)
            ks = [t - t0 for t in ts[1:]]
            row.update(regret=round(analysis.regret(ks, g[1:]), 3), peak=round(1e3 * g[1:].max(), 2),
                       end=round(1e3 * g[-1], 2))
        rows.append(row)
    out = Path(e['report'])
    out.mkdir(parents=True, exist_ok=True)
    analysis.write_csv(rows, out / 'xdevice.csv')
    return rows
