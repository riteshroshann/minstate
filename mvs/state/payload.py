import time

import torch

from ..config import fingerprint
from . import codecs, wire
from .reconstruct import rebuild
from .schema import State, capture, frozen_id
from .strategy import get as strategy

COMPONENTS = ('theta', 'm', 'v')


class Clock:
    def __init__(self, device):
        self.device, self.ms = device, {}

    def __call__(self, key, fn, *args):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        out = fn(*args)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.ms[key] = self.ms.get(key, 0.0) + (time.perf_counter() - t0) * 1e3
        return out


def encode(state, strat):
    parts, shapes, sizes = {}, {}, dict.fromkeys(COMPONENTS + ('buffers',), 0)
    for comp in COMPONENTS:
        name = getattr(strat, comp)
        if name is None:
            continue
        codec = codecs.get(name)
        for n, x in getattr(state, comp).items():
            shapes[n] = list(x.shape)
            for key, t in codec.enc(x).items():
                parts[f'{comp}/{n}/{key}'] = t
                sizes[comp] += t.numel() * t.element_size()
    for n, b in state.buffers.items():
        parts[f'buffers/{n}/x'] = b.detach()
        sizes['buffers'] += b.numel() * b.element_size()
    return parts, shapes, sizes


def to_host(parts):
    pin = any(t.is_cuda for t in parts.values())
    out = {}
    for k, t in parts.items():
        out[k] = torch.empty(t.shape, dtype=t.dtype, pin_memory=pin).copy_(t, non_blocking=pin) if t.is_cuda else t
    return out


def to_device(tensors, device):
    return {k: t.to(device, non_blocking=True) for k, t in tensors.items()}


def decode(tensors, meta):
    strat, grouped, bufs = strategy(meta['strategy']), {c: {} for c in COMPONENTS}, {}
    for key, t in tensors.items():
        comp, rest = key.split('/', 1)
        n, part = rest.rsplit('/', 1)
        if comp == 'buffers':
            bufs[n] = t
        else:
            grouped[comp].setdefault(n, {})[part] = t
    out = {}
    for comp in COMPONENTS:
        name = getattr(strat, comp)
        codec = codecs.get(name) if name else None
        out[comp] = {n: codec.dec(p, meta['shapes'][n]) for n, p in grouped[comp].items()} if codec else None
    return State(out['theta'], out['m'], out['v'], bufs, meta['tau'], meta['step'], dict(meta['extra'])), strat


def metadata(tr, strat, state, shapes, sizes):
    return {'strategy': strat.name, 'step': state.t, 'tau': state.tau, 'extra': state.extra, 'shapes': shapes,
            'sizes': sizes, 'fingerprint': fingerprint(tr.cfg), 'frozen': frozen_id(tr.raw), 'world': tr.world,
            'params': tr.n_params(), 'time': time.time()}


def pack(tr, name, clock):
    strat = strategy(name)
    state = capture(tr)
    parts, shapes, sizes = clock('encode', encode, state, strat)
    host = clock('d2h', to_host, parts)
    meta = metadata(tr, strat, state, shapes, sizes)
    buf = clock('serialise', wire.dumps, meta, host)
    return buf, meta


def unpack(tr, buf, clock):
    meta, tensors = clock('deserialise', wire.loads, buf)
    warn = []
    if meta['fingerprint'] != fingerprint(tr.cfg):
        warn.append('configuration fingerprint differs')
    if meta['frozen'] != frozen_id(tr.raw):
        raise ValueError('frozen base model differs from the one the payload was trained against')
    dev = clock('h2d', to_device, tensors, tr.device)
    state, strat = clock('decode', decode, dev, meta)
    k = clock('rebuild', rebuild, tr, state, strat)
    return meta, k, warn


def stats(meta, clock, nbytes, extra=None):
    return {'strategy': meta['strategy'], 'step': meta['step'], 'bytes': sum(meta['sizes'].values()),
            'file_bytes': nbytes, 'sizes': meta['sizes'], 'ms': dict(clock.ms), **(extra or {})}


def save(tr, name, path):
    clock = Clock(tr.device)
    buf, meta = pack(tr, name, clock)
    clock('write', wire.write, path, buf)
    return stats(meta, clock, len(buf))


def load(tr, path):
    clock = Clock(tr.device)
    buf = clock('read', wire.read, path)
    meta, k, warn = unpack(tr, buf, clock)
    return stats(meta, clock, len(buf), {'passes': k, 'warnings': warn, 'source_world': meta['world']})


def migrate(src, dst, name):
    clock = Clock(src.device)
    buf, meta = pack(src, name, clock)
    meta, k, warn = unpack(dst, buf, clock)
    return stats(meta, clock, len(buf), {'passes': k, 'warnings': warn})
