import hashlib
import json
from dataclasses import asdict, dataclass, field, fields

import yaml


@dataclass
class Config:
    workload: str = 'shakespeare_char'
    data_dir: str = 'data'
    synthetic: int = 0
    model: str = 'gpt'
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.2
    bias: bool = False
    hf_model: str = 'NousResearch/Llama-2-7b-hf'
    hf_revision: str = 'main'
    quant: str = 'nf4'
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list = field(default_factory=lambda: ['q_proj', 'v_proj'])
    grad_ckpt: bool = True
    batch_size: int = 64
    micro_batches: int = 1
    max_steps: int = 2000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-8
    grad_clip: float = 1.0
    amp: str = 'bf16'
    seed: int = 1337
    deterministic: str = 'strict'
    device: str = 'auto'
    eval_every: int = 10
    eval_samples: int = 512
    eval_batch: int = 64
    out_dir: str = 'runs/default'
    snapshots: list = field(default_factory=list)
    ckpt_every: int = 0
    ckpt_strategy: str = 'w_mq8_vlog8'
    strategy: str = 'w_vr1'
    warm_k: int = 20
    rewarm_k: int = 100
    notice_file: str = ''
    aws_spot: bool = False
    gcp_spot: bool = False
    auto_resume: bool = False
    resume: str = ''
    save_final: bool = False
    crash_at: int = -1


RUNTIME = {'data_dir', 'device', 'eval_every', 'out_dir', 'snapshots', 'ckpt_every', 'ckpt_strategy', 'strategy',
           'notice_file', 'aws_spot', 'gcp_spot', 'auto_resume', 'resume', 'save_final', 'crash_at', 'deterministic'}


def coerce(key, value):
    kind = type(getattr(Config(), key))
    if kind is float and isinstance(value, (int, str)):
        return float(value)
    if kind is int and isinstance(value, str):
        return int(value)
    if kind is list and isinstance(value, str):
        return [v for v in value.split(',') if v]
    return value


def load(path=None, overrides=()):
    raw = {}
    if path:
        with open(path) as f:
            raw.update(yaml.safe_load(f) or {})
    for item in overrides:
        key, value = item.split('=', 1)
        raw[key.replace('-', '_')] = yaml.safe_load(value)
    known = {f.name for f in fields(Config)}
    bad = sorted(set(raw) - known)
    if bad:
        raise KeyError(f'unknown config keys: {bad}')
    return Config(**{k: coerce(k, v) for k, v in raw.items()})


def fingerprint(cfg):
    d = {k: v for k, v in asdict(cfg).items() if k not in RUNTIME}
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def dump(cfg, path):
    with open(path, 'w') as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
