import pytest
import torch

from mvs.state.reconstruct import estimate


def adam_updates(tau, n, m0=0.0, g=0.3, betas=(0.9, 0.99)):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1.0, betas=betas, eps=0.0, weight_decay=0.0)
    opt.state[p] = {'step': torch.tensor(float(tau)), 'exp_avg': torch.full((1,), m0), 'exp_avg_sq': torch.zeros(1)}
    out = []
    for _ in range(n):
        before = p.detach().clone()
        p.grad = torch.full((1,), g)
        opt.step()
        out.append((before - p.detach()).item())
    return out


def test_zeroed_moments_at_large_counter_overshoot():
    u = adam_updates(10_000, 30)
    assert max(u) == pytest.approx(2.13, abs=0.01) and u.index(max(u)) + 1 == 13
    assert all(x > 1 for x in u[1:])


def test_fresh_counter_gives_sign_steps():
    assert all(x == pytest.approx(1.0, abs=1e-6) for x in adam_updates(0, 20))


def test_installed_estimate_uses_history_weight():
    b1, tau, mbar, g = 0.9, 7, 0.5, 0.3
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=1.0, betas=(b1, 0.99), eps=0.0, weight_decay=0.0)
    opt.state[p] = {'step': torch.tensor(float(tau)), 'exp_avg': torch.full((1,), (1 - b1 ** tau) * mbar),
                    'exp_avg_sq': torch.full((1,), (1 - 0.99 ** tau) * g * g)}
    p.grad = torch.full((1,), g)
    opt.step()
    w = b1 * (1 - b1 ** tau) / (1 - b1 ** (tau + 1))
    mhat = opt.state[p]['exp_avg'].item() / (1 - b1 ** (tau + 1))
    assert mhat == pytest.approx(w * mbar + (1 - w) * g, rel=1e-6)


def test_estimator_matches_manual_ema(cfg):
    from mvs.trainer import Trainer
    from mvs.workloads import RECON
    tr = Trainer(cfg)
    tr.step()
    k, grads = 5, []
    for i in range(k):
        tr.grads(tr.t, RECON, sub=i + 1)
        grads.append({n: p.grad.clone() for n, p in tr.params.items()})
        tr.opt.zero_grad(set_to_none=True)
    m, v = estimate(tr, k)
    n = next(iter(tr.params))
    em = torch.zeros_like(grads[0][n])
    ev = torch.zeros_like(em)
    for g in grads:
        em = 0.9 * em + 0.1 * g[n]
        ev = 0.99 * ev + 0.01 * g[n] ** 2
    assert torch.allclose(m[n], em / (1 - 0.9 ** k), atol=1e-7) and torch.allclose(v[n], ev / (1 - 0.99 ** k), atol=1e-9)
