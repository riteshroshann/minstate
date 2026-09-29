from dataclasses import dataclass

from . import codecs

TABLE = {
    'full': ('fp32', 'fp32', 'fp32', 'none'),
    'w_mq8_vlog8': ('fp32', 'q8', 'log8', 'none'),
    'w_m_vr1': ('fp32', 'fp32', 'rank1', 'none'),
    'w_m_warm': ('fp32', 'fp32', None, 'warm'),
    'w_v': ('fp32', None, 'fp32', 'zeros'),
    'w_vr1': ('fp32', None, 'rank1', 'zeros'),
    'w_vsvd4': ('fp32', None, 'svd4', 'zeros'),
    'wbf16_vr1': ('bf16', None, 'rank1', 'zeros'),
    'w_warm': ('fp32', None, None, 'warm'),
    'w_warmv': ('fp32', None, None, 'warmv'),
    'w_fresh': ('fp32', None, None, 'fresh'),
    'w_rewarm': ('fp32', None, None, 'rewarm'),
    'noise': ('fp32', 'fp32', 'fp32', 'none'),
}
STUDY = [k for k in TABLE if k != 'noise']
RHO = {'none', 'zeros', 'warm', 'warmv', 'fresh', 'rewarm'}


@dataclass(frozen=True)
class Strategy:
    name: str
    theta: str
    m: str | None
    v: str | None
    rho: str

    def passes(self, cfg):
        return cfg.warm_k if self.rho in ('warm', 'warmv') else 0


def validate(s):
    if s.theta is None:
        raise ValueError('the weights must travel')
    if s.rho not in RHO:
        raise ValueError(f'unknown reconstruction {s.rho}')
    for c in (s.theta, s.m, s.v):
        if c:
            codecs.get(c)
    if s.m and s.v is None and s.rho != 'warm':
        raise ValueError('keeping m while zeroing v makes the first update ~10x too large (Lemma 3.9)')
    if s.rho == 'none' and (s.m is None or s.v is None):
        raise ValueError('a dropped moment needs a reconstruction')
    if s.rho in ('fresh', 'rewarm', 'warmv') and (s.m or s.v):
        raise ValueError(f'{s.rho} rebuilds both moments; send neither')
    return s


def get(name):
    if name in TABLE:
        spec = TABLE[name]
    else:
        parts = name.split(':')
        if len(parts) != 4:
            raise KeyError(f'unknown strategy {name}; use a name or theta:m:v:rho')
        spec = tuple(None if p in ('', '-', 'none') else p for p in parts[:3]) + (parts[3],)
    return validate(Strategy(name, *spec))
