import hashlib
import json
from dataclasses import dataclass, field

import torch


@dataclass
class State:
    theta: dict
    m: dict | None
    v: dict | None
    buffers: dict
    tau: int
    t: int
    extra: dict = field(default_factory=dict)


def named_buffers(model):
    own = getattr(model, 'mvs_buffers', None)
    if own:
        return own()
    keys = set(model.state_dict().keys())
    return {n: b for n, b in model.named_buffers() if n in keys}


def frozen_id(model):
    items = sorted((n, list(p.shape), str(p.dtype)) for n, p in model.named_parameters() if not p.requires_grad)
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()[:16] if items else ''


def capture(tr):
    st = tr.opt.state
    moment = lambda key: {n: st[p][key] if p in st else torch.zeros_like(p) for n, p in tr.params.items()}
    tau = next((int(st[p]['step']) for p in tr.params.values() if p in st), 0)
    return State({n: p.detach() for n, p in tr.params.items()}, moment('exp_avg'), moment('exp_avg_sq'),
                 {n: b.detach() for n, b in named_buffers(tr.raw).items()}, tau, tr.t, dict(tr.extra))


def check(tr, s):
    if set(s.theta) != set(tr.params):
        missing, extra = sorted(set(tr.params) - set(s.theta)), sorted(set(s.theta) - set(tr.params))
        raise KeyError(f'parameter sets differ: missing {missing[:3]} unexpected {extra[:3]}')


@torch.no_grad()
def install_weights(tr, s):
    check(tr, s)
    for n, p in tr.params.items():
        p.copy_(s.theta[n])
    for n, b in named_buffers(tr.raw).items():
        b.copy_(s.buffers[n])
    tr.t, tr.extra = s.t, dict(s.extra)


def fresh(p, x):
    return torch.empty_like(p, memory_format=torch.preserve_format).copy_(x)


def install_moments(tr, s):
    on_device = bool(tr.opt.defaults.get('fused')) or bool(tr.opt.defaults.get('capturable'))
    for n, p in tr.params.items():
        step = torch.tensor(float(s.tau), dtype=torch.float32, device=p.device if on_device else 'cpu')
        tr.opt.state[p] = {'step': step, 'exp_avg': fresh(p, s.m[n]), 'exp_avg_sq': fresh(p, s.v[n])}
