import math

import pytest
import torch

from mvs.policy import payload as analytic
from mvs.state import codecs
from mvs.state.payload import encode
from mvs.state.schema import capture
from mvs.state.strategy import STUDY, get
from mvs.trainer import Trainer

SHAPES = [(1000,), (37, 300), (256,), (8, 3, 3, 3)]


def blockmax(x):
    f = torch.nn.functional.pad(x.reshape(-1), (0, (-x.numel()) % 256)).view(-1, 256)
    return f.abs().amax(1, keepdim=True).expand(-1, 256).reshape(-1)[:x.numel()].view(x.shape)


@pytest.mark.parametrize('shape', SHAPES)
def test_q8_error_bound(shape):
    torch.manual_seed(0)
    x = torch.randn(shape) * torch.logspace(-3, 1, math.prod(shape)).view(shape)
    x.view(-1)[:256] = 0
    q8 = codecs.get('q8')
    err = (q8.dec(q8.enc(x), shape) - x).abs()
    assert (err <= blockmax(x) / 254 + 1e-7).all()


@pytest.mark.parametrize('shape', SHAPES)
def test_log8_error_bound_and_zeros(shape):
    torch.manual_seed(0)
    x = 10 ** torch.empty(shape).uniform_(-12, 0)
    x.view(-1)[::7] = 0
    c = codecs.get('log8')
    y = c.dec(c.enc(x), shape)
    assert (y[x == 0] == 0).all() and (y[x > 0] > 0).all()
    b = torch.nn.functional.pad(x.reshape(-1), (0, (-x.numel()) % 256)).view(-1, 256)
    lg = torch.where(b > 0, b.log(), float('nan'))
    span = (lg.nan_to_num(-1e9).amax(1) - lg.nan_to_num(1e9).amin(1)).clamp_min(0)
    bound = span[:, None].expand(-1, 256).reshape(-1)[:x.numel()].view(shape) / 508
    pos = x > 0
    assert ((y[pos].log() - x[pos].log()).abs() <= bound[pos] + 1e-4).all()


def test_rank1_preserves_sums_and_is_kl_optimal():
    torch.manual_seed(0)
    v = torch.rand(40, 70).pow(2) + 1e-6
    c = codecs.get('rank1')
    r = c.dec(c.enc(v), v.shape)
    assert torch.allclose(r.sum(1), v.sum(1), rtol=1e-4) and torch.allclose(r.sum(0), v.sum(0), rtol=1e-4)
    kl = lambda q: (v * (v / q).log() - v + q).sum()
    base = kl(r)
    for _ in range(20):
        q = r * torch.exp(0.05 * torch.randn(40, 1)) * torch.exp(0.05 * torch.randn(1, 70))
        assert kl(q) >= base - 1e-4


def test_rank1_sends_vectors_exactly():
    c, x = codecs.get('rank1'), torch.rand(128)
    assert torch.equal(c.dec(c.enc(x), x.shape), x)


def test_svd_exact_on_low_rank_and_non_negative():
    u, w = torch.rand(30, 1), torch.rand(1, 50)
    v = (u @ w).square()
    c = codecs.get('svd4')
    y = c.dec(c.enc(v), v.shape)
    assert (y >= 0).all() and torch.allclose(y, v, atol=1e-5)
    assert c.nbytes((30, 50)) == 4 * 4 * 80 and c.nbytes((3, 3)) == 36


def test_exact_codecs():
    x = torch.randn(10, 10)
    assert torch.equal(codecs.get('fp32').dec(codecs.get('fp32').enc(x), x.shape), x)
    assert codecs.get('bf16').nbytes((10, 10)) == 200
    with pytest.raises(KeyError):
        codecs.get('zip')


@pytest.mark.parametrize('name', STUDY)
def test_measured_bytes_equal_analytic(cfg, name):
    tr = Trainer(cfg)
    tr.step()
    _, _, sizes = encode(capture(tr), get(name))
    shapes = [tuple(p.shape) for p in tr.params.values()]
    assert {k: sizes[k] for k in ('theta', 'm', 'v')} == analytic(shapes, name)
