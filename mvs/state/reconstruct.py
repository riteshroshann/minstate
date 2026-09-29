import torch

from ..workloads import RECON
from .schema import install_moments, install_weights, named_buffers


def estimate(tr, k):
    c = tr.cfg
    saved = {n: b.clone() for n, b in named_buffers(tr.raw).items()}
    m = {n: torch.zeros_like(p) for n, p in tr.params.items()}
    v = {n: torch.zeros_like(p) for n, p in tr.params.items()}
    for i in range(k):
        tr.grads(tr.t, RECON, sub=i + 1)
        for n, p in tr.params.items():
            m[n].mul_(c.beta1).add_(p.grad, alpha=1 - c.beta1)
            v[n].mul_(c.beta2).addcmul_(p.grad, p.grad, value=1 - c.beta2)
        tr.opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for n, b in named_buffers(tr.raw).items():
            b.copy_(saved[n])
    return {n: x / (1 - c.beta1 ** k) for n, x in m.items()}, {n: x / (1 - c.beta2 ** k) for n, x in v.items()}


def rebuild(tr, s, strat):
    c, k = tr.cfg, strat.passes(tr.cfg)
    install_weights(tr, s)
    zeros = lambda: {n: torch.zeros_like(p) for n, p in tr.params.items()}
    if k:
        mb, vb = estimate(tr, k)
        if s.m is None:
            s.m = {n: (1 - c.beta1 ** s.tau) * x for n, x in mb.items()} if strat.rho == 'warm' else zeros()
        if s.v is None:
            s.v = {n: (1 - c.beta2 ** s.tau) * x for n, x in vb.items()}
    if strat.rho in ('fresh', 'rewarm') or (s.m is None and s.v is None):
        s.m, s.v, s.tau = zeros(), zeros(), 0
    if s.m is None:
        s.m = zeros()
    if strat.rho == 'rewarm':
        s.extra['rewarm_t0'] = s.t
        tr.extra['rewarm_t0'] = s.t
    install_moments(tr, s)
    return k
