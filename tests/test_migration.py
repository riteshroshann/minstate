import importlib.util

import pytest
import torch

from mvs.config import load
from mvs.state import payload
from mvs.state.schema import named_buffers
from mvs.state.strategy import STUDY, get
from mvs.trainer import Trainer


def run(tr, n):
    return [tr.step() for _ in range(n)]


def params(tr):
    return {n: p.detach().clone() for n, p in tr.params.items()}


def test_trainers_are_bitwise_reproducible(cfg):
    a, b = Trainer(cfg), Trainer(cfg)
    assert run(a, 8) == run(b, 8)
    assert all(torch.equal(x, y) for x, y in zip(params(a).values(), params(b).values()))


def test_full_migration_is_bitwise_exact(cfg):
    a = Trainer(cfg)
    run(a, 15)
    b = Trainer(cfg)
    payload.migrate(a, b, 'full')
    assert b.t == a.t and run(a, 10) == run(b, 10) and a.evaluate() == b.evaluate()


def test_full_migration_through_a_file(cfg, tmp_path):
    a = Trainer(cfg)
    run(a, 12)
    payload.save(a, 'full', str(tmp_path / 'p.mvs'))
    b = Trainer(cfg)
    info = payload.load(b, str(tmp_path / 'p.mvs'))
    assert info['step'] == 12 and not info['warnings'] and run(a, 8) == run(b, 8)


def test_resnet_channels_last_and_batchnorm_are_exact(resnet_cfg):
    a = Trainer(resnet_cfg)
    run(a, 4)
    b = Trainer(resnet_cfg)
    payload.migrate(a, b, 'full')
    assert all(torch.equal(x, named_buffers(b.raw)[n]) for n, x in named_buffers(a.raw).items())
    p = next(v for v in b.params.values() if v.dim() == 4)
    assert b.opt.state[p]['exp_avg'].stride() == p.stride()
    assert run(a, 3) == run(b, 3)


def test_source_state_survives_repeated_migration(cfg):
    a = Trainer(cfg)
    run(a, 10)
    before = params(a)
    first, second = Trainer(cfg), Trainer(cfg)
    payload.migrate(a, first, 'full')
    losses = run(first, 3)
    payload.migrate(a, second, 'full')
    assert all(torch.equal(before[n], p) for n, p in a.params.items())
    assert run(second, 3) == losses


@pytest.mark.parametrize('name', STUDY)
def test_every_strategy_resumes(cfg, name):
    a = Trainer(cfg)
    run(a, 12)
    b = Trainer(cfg)
    info = payload.migrate(a, b, name)
    assert info['passes'] == get(name).passes(cfg)
    assert b.t == 12 and all(torch.isfinite(torch.tensor(run(b, 3))))
    tau = int(b.opt.state[next(iter(b.params.values()))]['step'])
    assert tau == (3 if name in ('w_fresh', 'w_rewarm') else 15)


def test_rewarm_scales_learning_rate(cfg):
    a = Trainer(cfg)
    run(a, 12)
    b = Trainer(cfg)
    payload.migrate(a, b, 'w_rewarm')
    assert b.lr_at(12) == pytest.approx(a.lr_at(12) / cfg.rewarm_k)


def test_warm_reconstruction_leaves_weights_and_buffers(resnet_cfg):
    a = Trainer(resnet_cfg)
    run(a, 3)
    b = Trainer(resnet_cfg)
    payload.migrate(a, b, 'w_warm')
    assert all(torch.equal(a.params[n], p) for n, p in b.params.items())
    assert all(torch.equal(x, named_buffers(b.raw)[n]) for n, x in named_buffers(a.raw).items())


def test_invalid_strategies_are_rejected():
    with pytest.raises(ValueError, match='Lemma 3.9'):
        get('fp32:fp32:-:zeros')
    with pytest.raises(ValueError):
        get('-:fp32:fp32:none')
    assert get('bf16:q8:rank1:none').v == 'rank1'


def test_parameter_mismatch_is_rejected(cfg, overrides, tmp_path):
    a = Trainer(cfg)
    payload.save(a, 'full', str(tmp_path / 'p.mvs'))
    other = Trainer(load(None, [*overrides, 'n_layer=3']))
    with pytest.raises(KeyError, match='parameter sets differ'):
        payload.load(other, str(tmp_path / 'p.mvs'))


@pytest.mark.skipif(importlib.util.find_spec('transformers') is None, reason='transformers not installed')
def test_qlora_adapters_migrate_exactly(data_dir):
    c = load(None, [f'data_dir={data_dir}', 'model=qlora', 'hf_model=tiny',
             'quant=none', 'lora_r=4', 'block_size=32', 'batch_size=4', 'micro_batches=2', 'amp=none',
             'max_steps=30', 'warmup=5', 'eval_samples=16', 'eval_batch=8', 'weight_decay=0.0'])
    a = Trainer(c)
    assert all('lora_' in n for n in a.params)
    run(a, 6)
    b = Trainer(c)
    info = payload.migrate(a, b, 'full')
    assert info['sizes']['theta'] == 4 * a.n_params()
    assert run(a, 4) == run(b, 4)
