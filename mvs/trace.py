import json
from pathlib import Path

import numpy as np
import torch
import yaml

from . import analysis, manifest
from .config import load
from .experiment import forks
from .state import payload
from .state.strategy import STUDY
from .trainer import Trainer


def spec(path):
    with open(path) as f:
        e = yaml.safe_load(f)
    e.setdefault('strategies', STUDY)
    e.setdefault('seeds', [1337])
    e.setdefault('steps', 150)
    return e


def flat(tr):
    return torch.cat([p.detach().flatten() for p in tr.params.values()])


def norms(tr, steps):
    """Size of each of the next `steps` parameter updates in units of the learning rate.  -> (steps,)"""
    out = []
    for _ in range(steps):
        before, lr = flat(tr), tr.lr_at(tr.t)
        tr.step()
        out.append(((flat(tr) - before).norm() / lr).item())
    return out


def predicted(rho, b1, b2, k):
    """Step inflation k steps after arrival for gradients that are pure noise; see the paper's transient section."""
    k = np.asarray(k, dtype=float)
    if rho == 'zeros':  # both moments zeroed, counter kept
        return np.sqrt((1 - b1 ** (2 * k)) / (1 - b2 ** k))
    if rho == 'fresh':  # both zeroed, counter reset: bias correction cancels the v deficit
        return np.sqrt((1 + b1 ** k) / (1 - b1 ** k))
    if rho == 'm_dropped':  # v intact, m refilling from zero
        return np.sqrt(1 - b1 ** (2 * k))
    raise KeyError(rho)


def run(path):
    e, rows = spec(path), []
    for scale, ov in e['scales'].items():
        for seed in e['seeds']:
            cfg = load(e['base'], [f'{k}={json.dumps(v)}' for k, v in {**(ov or {}), 'seed': seed}.items()])
            for t0 in forks(cfg, e['forks']):
                snap = str(Path(e['snaps']) / scale / f's{seed}' / 'control' / f'snap_{t0}.mvs')
                ref = Trainer(cfg)
                payload.load(ref, snap)
                base = norms(ref, e['steps'])
                for name in e['strategies']:
                    src, dst = Trainer(cfg, data=ref.data), Trainer(cfg, data=ref.data)
                    payload.load(src, snap)
                    payload.migrate(src, dst, name)
                    for k, (a, b) in enumerate(zip(norms(dst, e['steps']), base), 1):
                        rows.append({'scale': scale, 'seed': seed, 'fork': t0, 'strategy': name, 'k': k,
                                     'ratio': round(a / b, 5)})
                    first = [r['ratio'] for r in rows[-e['steps']:][:3]]
                    print(f'{scale} s{seed} fork {t0} {name:12s} first ratios {first}', flush=True)
    out = Path(e['report'])
    out.mkdir(parents=True, exist_ok=True)
    analysis.write_csv(rows, out / 'steps.csv')
    (out / 'steps.json').write_text(json.dumps({**manifest.environment(ref.device), 'spec': e}, indent=1))
    return rows
