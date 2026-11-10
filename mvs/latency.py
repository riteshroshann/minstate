import json
from pathlib import Path

import numpy as np
import yaml

from . import analysis, manifest
from .config import load
from .experiment import forks
from .state import payload, wire
from .state.strategy import STUDY
from .trainer import Trainer

DRAIN = ('encode', 'd2h', 'serialise', 'write')
RESTORE = ('read', 'deserialise', 'h2d', 'decode', 'rebuild')


def spec(path):
    with open(path) as f:
        e = yaml.safe_load(f)
    e.setdefault('seed', 1337)
    e.setdefault('strategies', STUDY)
    e.setdefault('reps', 20)
    e.setdefault('warmup', 3)
    return e


def once(src, dst, name, path):
    clock = payload.Clock(src.device)
    buf, meta = payload.pack(src, name, clock)
    clock('write', wire.write, path, buf)
    back = clock('read', wire.read, path)
    payload.unpack(dst, back, clock)
    return {'bytes': len(buf), **clock.ms}


def run(path, device):
    e, rows = spec(path), []
    out = Path(e['report'])
    out.mkdir(parents=True, exist_ok=True)
    for scale, ov in e['scales'].items():
        cfg = load(e['base'], [f'{k}={json.dumps(v)}' for k, v in {**(ov or {}), 'seed': e['seed'], 'device': device}.items()])
        t0 = forks(cfg, [e['fork']])[0]
        src, dst = Trainer(cfg), Trainer(cfg)
        payload.load(src, str(Path(e['snaps']) / scale / f"s{e['seed']}" / 'control' / f'snap_{t0}.mvs'))
        tmp = out / f'_{device}.mvs'
        for name in e['strategies']:
            # the first few round trips pay for allocator and page-cache warm-up, so they are run and dropped
            for rep in range(-e['warmup'], e['reps']):
                r = once(src, dst, name, str(tmp))
                if rep >= 0:
                    rows.append({'scale': scale, 'params': src.n_params(), 'device': device, 'strategy': name, 'rep': rep,
                                 **r, 'drain': sum(r.get(k, 0.0) for k in DRAIN),
                                 'restore': sum(r.get(k, 0.0) for k in RESTORE)})
            sel = [r for r in rows if r['scale'] == scale and r['strategy'] == name]
            print(f"{device} {scale} {name:12s} {sel[0]['bytes'] / 1e6:8.2f} MB  drain {np.median([r['drain'] for r in sel]):8.1f} ms"
                  f"  restore {np.median([r['restore'] for r in sel]):8.1f} ms", flush=True)
        tmp.unlink(missing_ok=True)
    analysis.write_csv(rows, out / f'latency_{device}.csv')
    (out / f'latency_{device}.json').write_text(json.dumps(manifest.environment(src.device), indent=1))
    return rows


def summarise(rows, keys=(*DRAIN, *RESTORE, 'drain', 'restore')):
    """Median with 5th and 95th percentiles over repetitions, per (scale, device, strategy)."""
    out, groups = [], {}
    for r in rows:
        groups.setdefault((r['scale'], r['device'], r['strategy']), []).append(r)
    for (scale, device, name), sel in groups.items():
        row = {'scale': scale, 'device': device, 'strategy': name, 'params': sel[0]['params'], 'bytes': sel[0]['bytes'],
               'reps': len(sel)}
        for k in keys:
            x = np.array([float(r.get(k) or 0.0) for r in sel])
            row.update({k: round(float(np.median(x)), 3), f'{k}_p5': round(float(np.percentile(x, 5)), 3),
                        f'{k}_p95': round(float(np.percentile(x, 95)), 3)})
        out.append(row)
    return out
