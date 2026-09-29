import math
import re

import torch
import torch.nn.functional as F

NB = 256


def numel(shape):
    return math.prod(shape)


def matrix(shape):
    return (shape[0], numel(shape[1:])) if len(shape) >= 2 else None


def blocks(x):
    f = x.detach().reshape(-1).float()
    return F.pad(f, (0, (-f.numel()) % NB)).view(-1, NB)


def unblock(q, n):
    return F.pad(q.float(), (0, (-n) % NB)).view(-1, NB)


class Exact:
    def __init__(self, name, dtype):
        self.name, self.dtype = name, dtype

    def enc(self, x):
        return {'x': x.detach().to(self.dtype)}

    def dec(self, p, shape):
        return p['x'].float().view(shape)

    def nbytes(self, shape):
        return numel(shape) * torch.empty((), dtype=self.dtype).element_size()


class Raw:
    name = 'raw'

    def enc(self, x):
        return {'x': x.detach()}

    def dec(self, p, shape):
        return p['x'].view(shape)

    def nbytes(self, shape, dtype=torch.float32):
        return numel(shape) * torch.empty((), dtype=dtype).element_size()


class Q8:
    name = 'q8'

    def enc(self, x):
        b = blocks(x)
        s = b.abs().amax(1) / 127
        q = (b / torch.where(s > 0, s, 1)[:, None]).round().clamp(-127, 127).to(torch.int8)
        return {'q': q.view(-1)[:x.numel()], 's': s}

    def dec(self, p, shape):
        n = numel(shape)
        return (unblock(p['q'], n) * p['s'][:, None]).view(-1)[:n].view(shape)

    def nbytes(self, shape):
        n = numel(shape)
        return n + 4 * math.ceil(n / NB)


class Log8:
    name = 'log8'

    def enc(self, x):
        b = blocks(x)
        pos = b > 0
        lg = torch.where(pos, b.clamp_min(torch.finfo(torch.float32).tiny).log(), 0.0)
        lo = torch.where(pos, lg, math.inf).amin(1)
        hi = torch.where(pos, lg, -math.inf).amax(1)
        lo, hi = torch.where(lo.isfinite(), lo, 0.0), torch.where(hi.isfinite(), hi, 0.0)
        span = torch.where(hi > lo, hi - lo, 1.0)[:, None]
        q = torch.where(pos, 1 + (254 * (lg - lo[:, None]) / span).round(), 0.0)
        return {'q': q.clamp(0, 255).to(torch.uint8).view(-1)[:x.numel()], 'l': lo, 'h': hi}

    def dec(self, p, shape):
        n = numel(shape)
        q, lo, hi = unblock(p['q'], n), p['l'][:, None], p['h'][:, None]
        x = torch.where(q > 0, torch.exp(lo + (q - 1) * (hi - lo) / 254), 0.0)
        return x.view(-1)[:n].view(shape)

    def nbytes(self, shape):
        n = numel(shape)
        return n + 8 * math.ceil(n / NB)


class Rank1:
    name = 'rank1'

    def enc(self, x):
        if x.dim() < 2:
            return {'x': x.detach().float()}
        v = x.detach().float().reshape(x.shape[0], -1)
        return {'r': v.sum(1), 'c': v.sum(0)}

    def dec(self, p, shape):
        if 'x' in p:
            return p['x'].float().view(shape)
        r, c = p['r'], p['c']
        s = r.sum()
        return (torch.outer(r, c) / torch.where(s > 0, s, 1.0)).view(shape)

    def nbytes(self, shape):
        m = matrix(shape)
        return 4 * (m[0] + m[1]) if m else 4 * numel(shape)


class SVD:
    def __init__(self, k):
        self.k, self.name = k, f'svd{k}'

    def fits(self, shape):
        m = matrix(shape)
        return bool(m) and self.k * (m[0] + m[1]) < m[0] * m[1]

    def enc(self, x):
        if not self.fits(tuple(x.shape)):
            return {'x': x.detach().float()}
        root = x.detach().float().reshape(x.shape[0], -1).clamp_min(0).sqrt()
        u, s, vh = torch.linalg.svd(root, full_matrices=False)
        return {'u': (u[:, :self.k] * s[:self.k]).contiguous(), 'w': vh[:self.k].t().contiguous()}

    def dec(self, p, shape):
        if 'x' in p:
            return p['x'].float().view(shape)
        return (p['u'] @ p['w'].t()).square().view(shape)

    def nbytes(self, shape):
        m = matrix(shape)
        return 4 * self.k * (m[0] + m[1]) if self.fits(tuple(shape)) else 4 * numel(shape)


FIXED = {'fp32': Exact('fp32', torch.float32), 'bf16': Exact('bf16', torch.bfloat16),
         'q8': Q8(), 'log8': Log8(), 'rank1': Rank1(), 'raw': Raw()}


def get(name):
    if name in FIXED:
        return FIXED[name]
    m = re.fullmatch(r'svd(\d+)', name or '')
    if m:
        return SVD(int(m.group(1)))
    raise KeyError(f'unknown codec {name}')
