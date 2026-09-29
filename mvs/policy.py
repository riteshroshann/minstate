import math

from .analysis import read_csv
from .state import codecs
from .state.strategy import STUDY, get


def llama(layers=32, d=4096, ffn=11008, vocab=32000, kv=None, tied=False):
    kv = kv or d
    block = [(d, d), (kv, d), (kv, d), (d, d), (ffn, d), (ffn, d), (d, ffn), (d,), (d,)]
    return [s for _ in range(layers) for s in block] + [(vocab, d), (d,)] + ([] if tied else [(vocab, d)])


def lora(layers=32, d=4096, r=16, targets=2):
    return [s for _ in range(layers * targets) for s in ((r, d), (d, r))]


MODELS = {
    'llama-7b': llama(),
    'llama-13b': llama(40, 5120, 13824),
    'llama-7b-qlora': lora(),
    'gpt-xs': None, 'gpt-s': None, 'gpt-m': None,
}


def shapes(model):
    if MODELS.get(model):
        return MODELS[model]
    import torch
    from .config import Config
    from .models.gpt import GPT
    dims = {'gpt-xs': (4, 4, 128), 'gpt-s': (6, 6, 384), 'gpt-m': (8, 8, 512)}[model]
    cfg = Config(n_layer=dims[0], n_head=dims[1], n_embd=dims[2])
    with torch.device('meta'):
        net = GPT(cfg, 65)
    return [tuple(p.shape) for p in net.parameters()]


def payload(shape_list, name):
    s = get(name)
    return {c: sum(codecs.get(getattr(s, c)).nbytes(sh) for sh in shape_list) if getattr(s, c) else 0
            for c in ('theta', 'm', 'v')}


def steps_lost(rows, name, params, frac):
    fracs = sorted({float(r['frac']) for r in rows})
    f = min(fracs, key=lambda x: abs(x - frac))
    pts = sorted((float(r['params']), float(r['lam'])) for r in rows if r['strategy'] == name and float(r['frac']) == f)
    if not pts:
        return None, True
    if params <= pts[0][0]:
        return pts[0][1], params < pts[0][0]
    if params >= pts[-1][0]:
        return pts[-1][1], params > pts[-1][0]
    for (p0, l0), (p1, l1) in zip(pts, pts[1:]):
        if p0 <= params <= p1:
            w = (math.log(params) - math.log(p0)) / (math.log(p1) - math.log(p0))
            return l0 + w * (l1 - l0), False


def cost(nbytes, lam, gbps, step_s, gpus, price, egress, notice, ckpt_every):
    xfer = nbytes / (gbps * 1.25e8)
    rate = gpus * price / 3600
    if notice and xfer > notice:
        return {'fits': False, 'xfer': xfer, 'egress': 0.0, 'time': rate * ckpt_every / 2 * step_s,
                'total': rate * ckpt_every / 2 * step_s}
    eg, tm = egress * nbytes / 1e9, rate * (xfer + lam * step_s)
    return {'fits': True, 'xfer': xfer, 'egress': eg, 'time': tm, 'total': eg + tm}


def decide(model, summary, frac=0.6, gbps=10.0, step_s=6.0, gpus=8, price=1.2, egress=0.02, notice=30.0,
           ckpt_every=1000, src_price=None, remaining=None, strategies=None):
    sh, rows = shapes(model), read_csv(summary)
    params = sum(math.prod(s) for s in sh)
    table = []
    for name in strategies or STUDY:
        nbytes = sum(payload(sh, name).values())
        lam, extrapolated = steps_lost(rows, name, params, frac)
        if lam is None:
            continue
        table.append({'strategy': name, 'GB': nbytes / 1e9, 'lam': lam, 'extrapolated': extrapolated or 'qlora' in model,
                      **cost(nbytes, lam, gbps, step_s, gpus, price, egress, notice, ckpt_every)})
    table.sort(key=lambda r: (r['total'], r['lam']))
    out = {'model': model, 'params': params, 'table': table, 'choice': table[0]['strategy']}
    if src_price is not None and remaining is not None:
        per_step = (src_price - price) * gpus * step_s / 3600
        out['saving'] = per_step * remaining
        out['move'] = out['saving'] > table[0]['total']
        out['break_even_steps'] = math.ceil(table[0]['total'] / per_step) if per_step > 0 else None
    return out


def render(d):
    size = f"{d['params'] / 1e9:.2f}B" if d['params'] >= 1e9 else f"{d['params'] / 1e6:.1f}M"
    lines = [f"model {d['model']}  ({size} trainable params)",
             '| strategy | GB | transfer (s) | fits | steps lost | egress $ | time $ | total $ |', '|---|---|---|---|---|---|---|---|']
    for r in d['table']:
        mark = '†' if r['extrapolated'] else ''
        gb = f"{r['GB']:.1f}" if r['GB'] >= 1 else f"{r['GB']:.4f}"
        lines.append(f"| {r['strategy']} | {gb} | {r['xfer']:.2f} | {'yes' if r['fits'] else 'no'} | "
                     f"{r['lam']:.1f}{mark} | {r['egress']:.2f} | {r['time']:.2f} | {r['total']:.2f} |")
    lines.append(f"choice: {d['choice']}")
    if 'move' in d:
        lines.append(f"opportunistic: saving ${d['saving']:.2f} -> {'migrate' if d['move'] else 'stay'}; "
                     f"break-even {d['break_even_steps']} remaining steps")
    return '\n'.join(lines)
