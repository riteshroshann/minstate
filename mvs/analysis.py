import csv
import itertools
import json
from pathlib import Path

import numpy as np

from .state.strategy import TABLE

FAMILIES = {'moments carried': ('full', 'w_mq8_vlog8', 'w_m_vr1', 'w_m_warm'),
            'm dropped': ('w_v', 'w_vr1', 'w_vsvd4', 'wbf16_vr1', 'w_emb_v'),
            'both dropped': ('w_warm', 'w_warmv', 'w_fresh', 'w_rewarm')}
COLORS = {'moments carried': '#2f6fd6', 'm dropped': '#e8663a', 'both dropped': '#1fa67a'}
K = 50


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
    # how many steps behind the control the run is, read off the control's smoothed loss curve
    c = envelope(ctrl, w)
    d = running_mean(np.asarray(run, float) - np.asarray(ctrl[t0:t0 + len(run)], float), w2)
    return np.array([t - inverse(c, c[t] + d[k], t) for k, t in enumerate(range(t0, t0 + len(run)))])


def steps_lost(delta, passes=0):
    tail = float(np.mean(delta[int(0.75 * len(delta)):]))
    return passes + max(0.0, tail), tail


def seed_band(curves, t0, w):
    """eps: 95th percentile of pairwise |L_i(t) - L_j(t)| over held-out evals in (t0, t0 + w]."""
    ts = sorted(set.intersection(*(set(c) for c in curves)))
    ts = [t for t in ts if t0 < t <= t0 + w]
    gaps = [abs(a[t] - b[t]) for a, b in itertools.combinations(curves, 2) for t in ts]
    return float(np.percentile(gaps, 95)) if gaps else None


def str_steps(ks, gaps, eps, k=K):
    """Steps-to-Recovery: first offset whose next k steps all stay within eps of the control; None if censored."""
    if eps is None:
        return None
    # the weights always travel, so at the fork itself the held-out loss equals the control's
    ks, gaps = [0, *ks], [0.0, *gaps]
    last = ks[-1]
    for i, start in enumerate(ks):
        if start + k > last:
            return None
        if all(g <= eps for kk, g in zip(ks[i:], gaps[i:]) if kk <= start + k):
            return start
    return None


def regret(ks, gaps):
    # held-out loss is sampled every few steps, so each sample stands for the steps since the previous one
    width = np.diff([0, *ks])
    return float(sum(w * max(0.0, g) for w, g in zip(width, gaps)))


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


def seeds_of(scale_dir):
    return sorted(d for d in scale_dir.iterdir() if (d / 'control' / 'done.json').exists())


def band_rows(scale, params, ctrls, t0, w, eps):
    """The seed band as rows: each control, shifted against every other, is what pure noise looks like."""
    lams = [steps_lost(delays(a[0], b[0][t0:t0 + w], t0))[1] for a, b in itertools.permutations(ctrls, 2)]
    if not lams:
        return []
    return [{'scale': scale, 'params': params, 'seed': 'band', 'fork': t0, 'strategy': 'seed_band',
             'eps': round(eps * 1e3, 3) if eps is not None else '', 'lam': round(float(np.percentile(np.abs(lams), 95)), 2)}]


def rows_for(scale_dir):
    seeds = seeds_of(scale_dir)
    ctrls = {d.name: curves(d / 'control' / 'log.jsonl') for d in seeds}
    meta = json.loads((seeds[0] / 'control' / 'done.json').read_text())
    out = []
    runs = {d.name: [json.loads(p.read_text()) for p in sorted((d / 'migrations').glob('*.json'))] for d in seeds}
    forks = sorted({r['t0'] for rs in runs.values() for r in rs})
    for t0 in forks:
        w = max(len(r['train']) for rs in runs.values() for r in rs if r['t0'] == t0)
        eps = seed_band([c[1] for c in ctrls.values()], t0, w) if len(ctrls) > 1 else None
        out += band_rows(scale_dir.name, meta['params'], list(ctrls.values()), t0, w, eps)
        for seed, rs in runs.items():
            ctrl, cval = ctrls[seed]
            group = {r['strategy']: r for r in rs if r['t0'] == t0}
            full = group['full']['bytes'] if 'full' in group else max(r['bytes'] for r in group.values())
            for name, r in group.items():
                ks, gaps = val_gaps(r, t0, cval)
                lam, tail = steps_lost(delays(ctrl, r['train'], t0), r['passes'])
                rec = str_steps(ks, gaps, eps)
                out.append({'scale': scale_dir.name, 'params': meta['params'], 'seed': seed[1:], 'fork': t0,
                            'frac': round(t0 / meta['steps'], 3), 'strategy': name, 'MB': round(r['file_bytes'] / 1e6, 3),
                            'payload': round(r['bytes'] / full, 4), 'K': r['passes'],
                            'eps': round(eps * 1e3, 3) if eps is not None else '',
                            'StR': '' if eps is None else (rec if rec is not None else f'>{ks[-1]}'),
                            'regret': round(regret(ks, gaps), 3), 'lam': round(lam, 2), 'delta': round(tail, 2),
                            'peak': round(max(gaps) * 1e3, 2), 'end': round(gaps[-1] * 1e3, 2),
                            **{f'{k}_MB': round(v / 1e6, 4) for k, v in r['sizes'].items()},
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


def runs_only(rows):
    return [r for r in rows if r['strategy'] != 'seed_band']


def scales_of(rows):
    return sorted({r['scale'] for r in rows}, key=lambda s: float(next(r['params'] for r in rows if r['scale'] == s)))


def censored(r):
    return str(r['StR']).startswith('>')


def str_cell(sel):
    vals = [r['StR'] for r in sel if r['StR'] != '']
    if not vals:
        return '-'
    done = sorted(float(v) for v in vals if not str(v).startswith('>'))
    cens = len(vals) - len(done)
    med = f'{np.median(done):.0f}' if done else '>W'
    return med + (f' ({cens}/{len(vals)} >W)' if cens else '')


def table(rows):
    """Steps lost (mean ± sd over seeds and forks) per scale, then StR and regret pooled over scales."""
    band = [r for r in rows if r['strategy'] == 'seed_band']
    rows = runs_only(rows)
    scales = scales_of(rows)
    names = [n for n in TABLE if any(r['strategy'] == n for r in rows)]
    full = {(r['scale'], r['seed'], r['fork']): float(r['MB']) for r in rows if r['strategy'] == 'full'}
    head = '| strategy | payload | ' + ' | '.join(f'{s} steps lost' for s in scales) + ' | StR median | regret |'
    lines = [head, '|' + '---|' * (len(scales) + 4)]
    for n in names:
        sel = [r for r in rows if r['strategy'] == n]
        pay = np.mean([float(r['MB']) / full[(r['scale'], r['seed'], r['fork'])] for r in sel])
        per = []
        for s in scales:
            v = [float(r['lam']) for r in sel if r['scale'] == s]
            per.append(f'{np.mean(v):.1f} ± {np.std(v):.1f}' if len(v) > 1 else (f'{v[0]:.1f}' if v else '-'))
        reg = np.mean([float(r['regret']) for r in sel])
        lines.append(f'| {n} | {pay:.0%} | ' + ' | '.join(per) + f' | {str_cell(sel)} | {reg:.2f} |')
    if band:
        per = [f"{np.mean([float(r['lam']) for r in band if r['scale'] == s]):.1f}" if any(r['scale'] == s for r in band)
               else '-' for s in scales]
        eps = [float(r['eps']) for r in band if r['eps'] != '']
        tail = f' | eps {np.mean(eps):.1f} mnats | - |' if eps else ' | - | - |'
        lines.append('| seed band (95th pct) | - | ' + ' | '.join(per) + tail)
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
    rows = runs_only(rows)
    scales = scales_of(rows)
    fam = {n: f for f, ns in FAMILIES.items() for n in ns}

    fig, ax = plt.subplots(figsize=(6, 3.2))
    for s in scales:
        vals = [curves(d / 'control' / 'log.jsonl')[1] for d in seeds_of(Path(root) / s)]
        ts = sorted(set.intersection(*(set(v) for v in vals)))
        ys = np.array([[v[t] for t in ts] for v in vals])
        line, = ax.plot(ts, ys.mean(0), lw=1.3, label=f'{s} ({len(vals)} seeds)')
        ax.fill_between(ts, ys.min(0), ys.max(0), color=line.get_color(), alpha=0.2, lw=0)
    ax.set(xlabel='step', ylabel='held-out loss (nats)')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'controls.pdf')

    for metric, label, fname in (('lam', 'steps lost', 'frontier.pdf'), ('regret', 'regret (nats x steps)', 'regret.pdf')):
        fig, axes = plt.subplots(1, len(scales), figsize=(3 * len(scales), 2.8), sharey=True, squeeze=False)
        for ax, s in zip(axes[0], scales):
            pts = []
            for n in sorted({r['strategy'] for r in rows if r['scale'] == s}):
                sel = [r for r in rows if r['scale'] == s and r['strategy'] == n]
                y = [float(r[metric]) for r in sel]
                pts.append((np.mean([float(r['payload']) for r in sel]), np.mean(y), n))
                ax.errorbar(pts[-1][0], pts[-1][1], yerr=np.std(y) / max(1, np.sqrt(len(y))), fmt='o', ms=3.5,
                            color=COLORS[fam[n]], elinewidth=0.8)
            f = frontier(pts)
            ax.step([p[0] for p in f], [p[1] for p in f], where='post', color='#888', lw=1)
            for x, y, n in f:
                ax.annotate(n, (x, y), fontsize=6, xytext=(3, 3), textcoords='offset points', family='monospace')
            ax.set(title=s, xlabel='payload / full')
        axes[0][0].set_ylabel(label)
        fig.tight_layout()
        fig.savefig(out / fname)

    big = scales[-1]
    sel = {r['strategy']: r for r in rows if r['scale'] == big}
    sel = sorted(sel.values(), key=lambda r: list(TABLE).index(r['strategy']), reverse=True)
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
    rows = [r for d in sorted(root.iterdir()) if d.is_dir() and seeds_of(d) for r in rows_for(d)]
    write_csv(rows, out / 'summary.csv')
    (out / 'table.md').write_text(table(rows) + '\n')
    figures(root, rows, out)
    return rows
