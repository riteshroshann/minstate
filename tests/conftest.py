import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mvs.config import load

WORDS = 'the king queen lord love death night day sword crown heart blood fair sweet noble speak thou thee my'.split()
TINY = ['n_layer=2', 'n_head=2', 'n_embd=32', 'block_size=32', 'batch_size=8', 'amp=none', 'max_steps=60',
        'warmup=10', 'eval_samples=32', 'eval_batch=16', 'micro_batches=2', 'dropout=0.1', 'eval_every=10']


@pytest.fixture(scope='session')
def data_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp('data')
    (root / 'shakespeare_char').mkdir()
    rng = random.Random(0)
    lines = (' '.join(rng.choice(WORDS) for _ in range(12)) for _ in range(3000))
    (root / 'shakespeare_char' / 'input.txt').write_text('\n'.join(lines))
    return str(root)


@pytest.fixture
def overrides(data_dir):
    return [f'data_dir={data_dir}', *TINY]


@pytest.fixture
def cfg(overrides):
    return load(None, overrides)


@pytest.fixture
def resnet_cfg(data_dir):
    return load(None, [f'data_dir={data_dir}', 'workload=cifar10', 'model=resnet18', 'synthetic=64', 'batch_size=8',
                       'eval_samples=16', 'eval_batch=8', 'amp=none', 'max_steps=20', 'warmup=2', 'lr=2e-3'])
