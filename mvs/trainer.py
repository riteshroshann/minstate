import contextlib
import math
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from . import models, workloads
from .workloads import TRAIN, mix

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def pick_device(cfg, local, world=1):
    # auto uses one GPU per rank; with fewer GPUs than ranks (or no NCCL) every rank falls back to CPU
    gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    fits = gpus >= world and (world == 1 or dist.is_nccl_available())
    kind = 'cuda' if cfg.device == 'auto' and fits else cfg.device
    kind = 'cpu' if kind == 'auto' else kind
    return torch.device('cuda', local) if kind.startswith('cuda') else torch.device(kind)


def determinism(cfg):
    on = cfg.deterministic != 'off'
    torch.use_deterministic_algorithms(on, warn_only=cfg.deterministic == 'warn')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = on
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


class Trainer:
    def __init__(self, cfg, rank=0, world=1, local=0, data=None):
        determinism(cfg)
        if cfg.micro_batches % world:
            raise ValueError(f'world size {world} must divide micro_batches {cfg.micro_batches}')
        self.cfg, self.rank, self.world, self.accum = cfg, rank, world, cfg.micro_batches // world
        self.device = pick_device(cfg, local, world)
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        self.data = data or workloads.build(cfg, self.device)
        self.raw = self.model = models.build(cfg, self.data.vocab, self.device)
        if world > 1:
            frozen = [n for n, p in self.raw.named_parameters() if not p.requires_grad]
            if frozen:
                DDP._set_params_and_buffers_to_ignore_for_model(self.raw, frozen)
            self.model = DDP(self.raw, device_ids=[self.device.index] if self.device.type == 'cuda' else None)
        self.params = {n: p for n, p in self.raw.named_parameters() if p.requires_grad}
        groups = [{'params': [p for p in self.params.values() if p.dim() >= 2], 'weight_decay': cfg.weight_decay},
                  {'params': [p for p in self.params.values() if p.dim() < 2], 'weight_decay': 0.0}]
        self.opt = torch.optim.AdamW([g for g in groups if g['params']], lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                                     eps=cfg.eps, fused=self.device.type == 'cuda')
        self.t, self.stream, self.extra = 0, TRAIN, {}

    def amp(self):
        if self.cfg.amp == 'bf16':
            return torch.autocast(self.device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def lr_at(self, t):
        c = self.cfg
        if t < c.warmup:
            lr = c.lr * (t + 1) / c.warmup
        else:
            r = min(1.0, (t - c.warmup) / max(1, c.max_steps - c.warmup))
            lr = c.min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (c.lr - c.min_lr)
        t0 = self.extra.get('rewarm_t0')
        return lr if t0 is None else lr * min(1.0, (t - t0 + 1) / c.rewarm_k)

    def loss(self, t, g, stream, sub=0):
        x, y = self.data.batch(t, g, stream, sub)
        torch.manual_seed(mix(self.cfg.seed, stream, t, g, sub, 7))
        with self.amp():
            return self.model(x, y)

    def grads(self, t, stream, sub=0):
        total = torch.zeros((), device=self.device)
        for j in range(self.accum):
            last = j == self.accum - 1 or self.world == 1
            with contextlib.nullcontext() if last else self.model.no_sync():
                loss = self.loss(t, self.rank * self.accum + j, stream, sub) / self.accum
                loss.backward()
            total += loss.detach()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(list(self.params.values()), self.cfg.grad_clip)
        return total

    def step(self):
        for group in self.opt.param_groups:
            group['lr'] = self.lr_at(self.t)
        loss = self.grads(self.t, self.stream)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        self.t += 1
        if self.world > 1:
            dist.all_reduce(loss)
            loss /= self.world
        return loss.item()

    @torch.no_grad()
    def evaluate(self):
        self.raw.eval()
        total, n = torch.zeros((), device=self.device), 0
        for x, y in self.data.evals():
            with self.amp():
                total += self.raw(x, y).float() * len(x)
            n += len(x)
        self.raw.train()
        return (total / n).item()

    def n_params(self):
        return sum(p.numel() for p in self.params.values())
