import csv
import json
from pathlib import Path

import numpy as np

from .state.strategy import TABLE

FAMILIES = {'moments carried': ('full', 'w_mq8_vlog8', 'w_m_vr1', 'w_m_warm'),
            'm dropped': ('w_v', 'w_vr1', 'w_vsvd4', 'wbf16_vr1'),
            'both dropped': ('w_warm', 'w_warmv', 'w_fresh', 'w_rewarm')}
COLORS = {'moments carried': '#2f6fd6', 'm dropped': '#e8663a', 'both dropped': '#1fa67a'}


def running_mean(x, w):
    x = np.asarray(x, float)
    c = np.concatenate([[0.0], np.cumsum(x)])
    i = np.arange(len(x))
    lo = np.maximum(0, i - w + 1)
    return (c[i + 1] - c[lo]) / (i + 1 - lo)


def envelope(x, w=50):
    return np.minimum.accumulate(running_mean(x, w))


def inverse(c, level, t):
    if level == c[t]:
        return float(t)
    if level > c[t]:
        s = t
        while s > 0 and c[s - 1] < level:
            s -= 1
        if s == 0:
            return 0.0
        a, b = c[s - 1], c[s]
        return s - 1 + (a - level) / (a - b) if a > b else float(s)
    s = t
    while s + 1 < len(c) and c[s + 1] > level:
        s += 1
    if s + 1 >= len(c):
        return float(len(c) - 1)
    a, b = c[s], c[s + 1]
    return s + (a - level) / (a - b) if a > b else float(s + 1)


def delays(ctrl, run, t0, w=50, w2=25):
    c = envelope(ctrl, w)
    d = running_mean(np.asarray(run, float) - np.asarray(ctrl[t0:t0 + len(run)], float), w2)
    return np.array([t - inverse(c, c[t] + d[k], t) for k, t in enumerate(range(t0, t0 + len(run)))])


def steps_lost(delta, passes=0):
    tail = float(np.mean(delta[int(0.75 * len(delta)):]))
    return passes + max(0.0, tail), tail


def noise_floor(gaps):
    return max(2e-3, max(abs(g) for g in gaps))


def recovered(ks, gaps, eps):
    bad = [k for k, g in zip(ks, gaps) if abs(g) > eps]
    if not bad:
        return 0
    return 'never' if bad[-1] == ks[-1] else next(k for k in ks if k > bad[-1])


def entries(path):
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def curves(path):
    train, val = {}, {}
    for e in entries(path):
        if 'loss' in e:
            train[e['t']] = e['loss']
        if 'val' in e:
            val[e['t']] = e['val']
    return [train[t] for t in sorted(train)], val


def val_gaps(r, t0, cval):
    ks = sorted(int(k) for k in r['val'])
    return [k - t0 for k in ks], [r['val'][str(k)] - cval[k] for k in ks]


def rows_for(scale_dir):
    ctrl, cval = curves(scale_dir / 'control' / 'log.jsonl')
    meta = json.loads((scale_dir / 'control' / 'done.json').read_text())
    runs = [json.loads(p.read_text()) for p in sorted((scale_dir / 'migrations').glob('*.json'))]
    out = []
    for t0 in sorted({r['t0'] for r in runs}):
        group = {r['strategy']: r for r in runs if r['t0'] == t0}
        eps = noise_floor(val_gaps(group['noise'], t0, cval)[1]) if 'noise' in group else 2e-3
        full = group['full']['bytes'] if 'full' in group else max(r['bytes'] for r in group.values())
        for name, r in group.items():
            ks, gaps = val_gaps(r, t0, cval)
            lam, tail = steps_lost(delays(ctrl, r['train'], t0), r['passes'])
            out.append({'scale': scale_dir.name, 'params': meta['params'], 'fork': t0,
                        'frac': round(t0 / meta['steps'], 3), 'strategy': name, 'MB': round(r['file_bytes'] / 1e6, 2), 'K': r['passes'],
                        'recovered': recovered(ks, gaps, eps) if name != 'noise' else '-', 'delta': round(tail, 2),
                        'lam': round(lam, 2), 'peak': round(max(gaps) * 1e3, 2), 'end': round(gaps[-1] * 1e3, 2),
                        'payload': round(r['bytes'] / full, 4), 'eps': round(eps * 1e3, 2),
                        **{f'{k}_MB': round(v / 1e6, 3) for k, v in r['sizes'].items()},
                        **{f'{k}_ms': round(v, 2) for k, v in r['ms'].items()}})
    return out


def write_csv(rows, path):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def table(rows):
    scales = sorted({r['scale'] for r in rows}, key=lambda s: float(next(r['params'] for r in rows if r['scale'] == s)))
    names = [n for n in TABLE if any(r['strategy'] == n for r in rows)]
    full = {(r['scale'], r['fork']): float(r['MB']) for r in rows if r['strategy'] == 'full'}
    head = '| strategy | payload | ' + ' | '.join(scales) + ' | mean | peak Δ |'
    lines = [head, '|' + '---|' * (len(scales) + 4)]
    for n in names:
        sel = [r for r in rows if r['strategy'] == n]
        per = [np.mean([float(r['lam']) for r in sel if r['scale'] == s]) for s in scales]
        pay = np.mean([float(r['MB']) / full[(r['scale'], r['fork'])] for r in sel])
        cells = [f'{pay:.0%}', *(f'{p:.1f}' for p in per), f'{np.mean(per):.1f}', f"{np.mean([float(r['peak']) for r in sel]):.1f}"]
        lines.append(f'| {n} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def frontier(points):
    best, out = float('inf'), []
    for x, y, n in sorted(points):
        if y < best:
            best = y
            out.append((x, y, n))
    return out


def figures(root, rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    scales = sorted({r['scale'] for r in rows}, key=lambda s: float(next(r['params'] for r in rows if r['scale'] == s)))
    fam = {n: f for f, ns in FAMILIES.items() for n in ns}
    fig, ax = plt.subplots(figsize=(6, 3.2))
    for s in scales:
        _, val = curves(Path(root) / s / 'control' / 'log.jsonl')
        ax.plot(sorted(val), [val[t] for t in sorted(val)], lw=1.3, label=s)
    ax.set(xlabel='step', ylabel='validation loss (nats)')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'controls.pdf')
    fig, axes = plt.subplots(1, len(scales), figsize=(3 * len(scales), 2.8), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], scales):
        pts = []
        for n in {r['strategy'] for r in rows if r['scale'] == s} - {'noise'}:
            sel = [r for r in rows if r['scale'] == s and r['strategy'] == n]
            pts.append((np.mean([float(r['payload']) for r in sel]), np.mean([float(r['lam']) for r in sel]), n))
        for x, y, n in pts:
            ax.scatter(x, y, s=14, color=COLORS[fam[n]])
        f = frontier(pts)
        ax.step([p[0] for p in f], [p[1] for p in f], where='post', color='#888', lw=1)
        for x, y, n in f:
            ax.annotate(n, (x, y), fontsize=6, xytext=(3, 3), textcoords='offset points', family='monospace')
        ax.set(title=s, xlabel='payload / full')
    axes[0][0].set_ylabel('steps lost')
    fig.tight_layout()
    fig.savefig(out / 'frontier.pdf')
    big = scales[-1]
    sel = [r for r in rows if r['scale'] == big and r['fork'] == max(q['fork'] for q in rows if q['scale'] == big)]
    sel = sorted((r for r in sel if r['strategy'] != 'noise'), key=lambda r: list(TABLE).index(r['strategy']), reverse=True)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    left = np.zeros(len(sel))
    for comp, color in (('theta', '#2f6fd6'), ('m', '#e8663a'), ('v', '#1fa67a')):
        w = np.array([float(r.get(f'{comp}_MB', 0) or 0) for r in sel])
        ax.barh([r['strategy'] for r in sel], w, left=left, color=color, label=comp)
        left += w
    ax.set(xlabel='payload (MB)', title=big)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out / 'payload.pdf')
    plt.close('all')


def report(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for d in sorted(root.iterdir()) if (d / 'control' / 'done.json').exists() for r in rows_for(d)]
    write_csv(rows, out / 'summary.csv')
    (out / 'table.md').write_text(table(rows) + '\n')
    figures(root, rows, out)
    return rows
