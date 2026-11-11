import json
import math
from pathlib import Path

import numpy as np
import yaml

from . import analysis, latency, policy, trace
from .config import load as load_config
from .manifest import ROOT
from .state.strategy import TABLE

# order and names used in every table and figure of the paper
ARMS = ['full', 'w_mq8_vlog8', 'w_m_vr1', 'w_v', 'w_vsvd4', 'w_emb_v', 'w_vr1', 'wbf16_vr1', 'w_m', 'w_zero', 'w_fresh',
        'w_rewarm', 'w_warmv', 'w_m_warm', 'w_warm']
CURVES = ['w_mq8_vlog8', 'w_vr1', 'wbf16_vr1', 'w_zero', 'w_fresh', 'w_warm', 'w_m']
COLORS = {'full': '#222222', 'w_mq8_vlog8': '#2f6fd6', 'w_vr1': '#e8663a', 'wbf16_vr1': '#b8451f', 'w_zero': '#1fa67a',
          'w_fresh': '#8a5cd0', 'w_warm': '#c23a6b', 'w_m': '#777777', 'w_v': '#d9a400'}
BOOT = 10000


def tex(name):
    return '\\texttt{' + name.replace('_', '\\_') + '}'


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def cluster_ci(by_seed, rng):
    """Mean and 95% interval, resampling whole seeds: the forks of one seed share a control and are not independent."""
    seeds = list(by_seed)
    mean = float(np.mean([v for s in seeds for v in by_seed[s]]))
    if len(seeds) < 2:
        return mean, None, None
    draws = rng.integers(0, len(seeds), (BOOT, len(seeds)))
    sums = np.array([np.sum(by_seed[s]) for s in seeds])
    counts = np.array([len(by_seed[s]) for s in seeds])
    means = sums[draws].sum(1) / counts[draws].sum(1)
    return mean, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def group(rows, key, scale=None, fork=None):
    out = {}
    for r in rows:
        if (scale is None or r['scale'] == scale) and (fork is None or int(r['fork']) == fork) and num(r[key]) is not None:
            out.setdefault(r['seed'], []).append(num(r[key]))
    return out


def fmt_ci(mean, lo, hi, d=1):
    return f'{mean:.{d}f}' if lo is None else f'{mean:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]'


def recovery(sel):
    """Cells that never left the band, median StR of the rest that recovered, and the censored count."""
    vals = [str(r['StR']) for r in sel if r['StR'] != '']
    zero = sum(v in ('0', '0.0') for v in vals)
    cens = sum(v.startswith('>') for v in vals)
    late = sorted(float(v) for v in vals if not v.startswith('>') and float(v) > 0)
    return len(vals), zero, (f'{np.median(late):.0f}' if late else '--'), cens


def arm_rows(rows, scale, rng, forks=None):
    runs = [r for r in analysis.runs_only(rows) if r['scale'] == scale and (forks is None or int(r['fork']) in forks)]
    eps = np.mean([num(r['eps']) for r in runs if num(r['eps']) is not None]) if runs else float('nan')
    out = []
    for name in ARMS:
        sel = [r for r in runs if r['strategy'] == name]
        if not sel:
            continue
        ratio = [{**r, 'ratio': num(r['peak']) / num(r['eps'])} for r in sel if num(r['eps'])]
        n, zero, late, cens = recovery(sel)
        out.append({'name': name, 'payload': 100 * np.mean([num(r['payload']) for r in sel]),
                    'peak': cluster_ci(group(sel, 'peak'), rng), 'ratio': cluster_ci(group(ratio, 'ratio'), rng),
                    'regret': cluster_ci(group(sel, 'regret'), rng), 'lam': cluster_ci(group(sel, 'lam'), rng),
                    'n': n, 'zero': zero, 'late': late, 'cens': cens, 'cells': len(sel),
                    'seeds': len({r['seed'] for r in sel})})
    return out, eps


def main_table(rows, scale, rng):
    arms, _ = arm_rows(rows, scale, rng)
    lines = []
    for a in arms:
        lines.append(' & '.join([tex(a['name']), f"{a['payload']:.1f}", fmt_ci(*a['peak'], d=1), fmt_ci(*a['ratio'], d=2),
                                 fmt_ci(*a['regret'], d=2), fmt_ci(*a['lam'], d=1), f"{a['zero']}/{a['n']}", a['late'],
                                 str(a['cens'])]) + ' \\\\')
    return '\n'.join(lines), arms


def scale_table(rows, rng):
    """Peak excess loss over eps per scale: is the picture stable as the model grows?"""
    scales, cols = analysis.scales_of(analysis.runs_only(rows)), {}
    for s in scales:
        cols[s] = {a['name']: a for a in arm_rows(rows, s, rng)[0]}
    lines = []
    for name in ARMS:
        if not any(name in cols[s] for s in scales):
            continue
        cells = []
        for s in scales:
            a = cols[s].get(name)
            cells += [fmt_ci(*a['ratio'], d=2), f"{a['zero']}/{a['n']}"] if a else ['--', '--']
        lines.append(' & '.join([tex(name), *cells]) + ' \\\\')
    return '\n'.join(lines), scales


def fork_table(rows, scale, rng):
    """Peak excess loss in mnats by migration point, with that point's eps in the header row."""
    runs = [r for r in analysis.runs_only(rows) if r['scale'] == scale]
    forks = sorted({int(r['fork']) for r in runs})
    eps = {f: np.mean([num(r['eps']) for r in runs if int(r['fork']) == f and num(r['eps']) is not None]) for f in forks}
    lines = ['$\\varepsilon$ (seed band) & ' + ' & '.join(f'{eps[f]:.1f}' for f in forks) + ' \\\\', '\\midrule']
    for name in ARMS:
        if name == 'full' or not any(r['strategy'] == name for r in runs):
            continue
        cells = []
        for f in forks:
            sel = [r for r in runs if r['strategy'] == name and int(r['fork']) == f]
            cells.append(fmt_ci(*cluster_ci(group(sel, 'peak'), rng), d=1) if sel else '--')
        lines.append(' & '.join([tex(name), *cells]) + ' \\\\')
    return '\n'.join(lines), forks


def bytes_table(rows, scale):
    """Measured bytes per parameter beside the analytic count from the codecs; they must agree (CLAUDE.md 6.5)."""
    runs = [r for r in analysis.runs_only(rows) if r['scale'] == scale]
    params = int(float(runs[0]['params']))
    shapes, lines, worst = policy.shapes(scale), [], 0.0
    for name in ARMS:
        sel = [r for r in runs if r['strategy'] == name]
        if not sel:
            continue
        measured = 1e6 * np.mean([num(r['MB']) for r in sel]) / params
        analytic = sum(policy.payload(shapes, name).values()) / params
        worst = max(worst, abs(measured - analytic) / analytic)
        spec = TABLE[name]
        lines.append(' & '.join([tex(name), *(s or '--' for s in spec[:3]), spec[3], f'{analytic:.3f}', f'{measured:.3f}',
                                 f'{100 * (measured - analytic) / analytic:+.2f}']) + ' \\\\')
    return '\n'.join(lines), worst


def latency_table(path, scale):
    rows = latency.summarise([{k: (num(v) if num(v) is not None else v) for k, v in r.items()}
                              for r in analysis.read_csv(path)])
    sel = {r['strategy']: r for r in rows if r['scale'] == scale}
    lines = []
    for name in ARMS:
        r = sel.get(name)
        if r:
            stage = lambda k: f"{r[k]:.1f}"
            both = lambda k: f"{r[k]:.1f} [{r[k + '_p5']:.1f}, {r[k + '_p95']:.1f}]"
            lines.append(' & '.join([tex(name), f"{r['bytes'] / 1e6:.1f}", stage('encode'), stage('d2h'), stage('serialise'),
                                     stage('write'), both('drain'), stage('read'), stage('h2d'), stage('decode'),
                                     stage('rebuild'), both('restore')]) + ' \\\\')
    return '\n'.join(lines), rows


def device_table(path):
    rows = analysis.read_csv(path)
    lines, worst = [], {}
    for r in rows:
        if r['strategy'] != 'full' or r.get('dev_max', '') == '':
            continue
        key = (r['device'], r['cond'])
        worst[key] = max(worst.get(key, 0.0), num(r['dev_max']) / num(r['eps']) if num(r['eps']) else 0.0)
        lines.append(' & '.join([r['scale'], r['fork'], r['device'], r['cond'].replace('_', '\\_'), r['arrival'], r['dev_mean'],
                                 r['dev_max'], f"{num(r['eps']):.1f}" if num(r['eps']) else '--',
                                 f"{num(r['dev_max']) / num(r['eps']):.2f}" if num(r['eps']) else '--']) + ' \\\\')
    return '\n'.join(lines), rows, worst


def diag_table(path, scale):
    """First update after arrival against the true one, mean over migration points, with the embedding class apart."""
    rows = [r for r in analysis.read_csv(path) if r['scale'] == scale]
    keys = ('cos', 'step_ratio', 'm_ratio', 'sqrt_v_err', 'matrix_cos', 'embedding_cos', 'matrix_sqrt_v_err',
            'embedding_sqrt_v_err')
    lines, by = [], {}
    for name in ARMS:
        sel = [r for r in rows if r['strategy'] == name]
        if not sel:
            continue
        by[name] = {k: float(np.mean([num(r[k]) for r in sel if num(r.get(k)) is not None] or [float('nan')])) for k in keys}
        a = by[name]
        lines.append(' & '.join([tex(name), f"{a['cos']:.2f}", f"{a['step_ratio']:.2f}", f"{a['m_ratio']:.2f}",
                                 f"{100 * a['sqrt_v_err']:.1f}", f"{a['matrix_cos']:.2f}", f"{a['embedding_cos']:.2f}",
                                 f"{100 * a['matrix_sqrt_v_err']:.1f}", f"{100 * a['embedding_sqrt_v_err']:.1f}"]) + ' \\\\')
    return '\n'.join(lines), by


def policy_rows(summary, scenarios):
    lines, picks = [], {}
    for label, kw in scenarios:
        d = policy.decide('llama-7b', summary, **kw)
        fits = [r['strategy'] for r in d['table'] if r['fits']]
        full = next(r for r in d['table'] if r['strategy'] == 'full')
        best = d['table'][0]
        picks[label] = best['strategy']
        lines.append(' & '.join([label, f"{kw['notice']:.0f}", f"{kw['gbps']:.0f}", f"{full['xfer']:.1f}",
                                 'yes' if full['fits'] else 'no', f"{len(fits)}/{len(d['table'])}", tex(best['strategy']),
                                 f"{best['GB']:.1f}", f"{best['xfer']:.1f}", f"{best['total']:.2f}"]) + ' \\\\')
    return '\n'.join(lines), picks


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': 0.6, 'lines.linewidth': 1.0, 'legend.frameon': False, 'pdf.fonttype': 42})
    return plt


def gap_curves(root, scale):
    """Per fork: offsets, and per arm the excess held-out loss in mnats for every seed."""
    out = {}
    for d in analysis.seeds_of(Path(root) / scale):
        cval = analysis.curves(d / 'control' / 'log.jsonl')[1]
        for p in sorted((d / 'migrations').glob('*.json')):
            r = json.loads(p.read_text())
            ks, gaps = analysis.val_gaps(r, r['t0'], cval)
            out.setdefault(r['t0'], {}).setdefault(r['strategy'], []).append((ks, 1e3 * np.array(gaps)))
    return out


def fig_curves(root, rows, scale, path):
    plt = style()
    data = gap_curves(root, scale)
    forks = sorted(data)
    fig, axes = plt.subplots(1, len(forks), figsize=(7.1, 2.1), sharey=True, squeeze=False)
    for ax, t0 in zip(axes[0], forks):
        eps = [num(r['eps']) for r in rows if r['scale'] == scale and r['strategy'] == 'seed_band' and int(r['fork']) == t0]
        if eps and eps[0] is not None:
            ax.axhspan(-eps[0], eps[0], color='#000000', alpha=0.07, lw=0)
        for name in CURVES:
            runs = data[t0].get(name)
            if not runs:
                continue
            n = min(len(g) for _, g in runs)
            ax.plot(runs[0][0][:n], np.mean([g[:n] for _, g in runs], 0), color=COLORS[name], label=name)
        ax.set_yscale('symlog', linthresh=10)
        ax.set_yticks([-10, 0, 10, 100, 1000], ['$-10$', '0', '10', '100', '1000'])
        ax.minorticks_off()
        ax.set_title(f'migrate at step {t0}', fontsize=8)
        ax.set_xlabel('steps after migration')
    axes[0][0].set_ylabel('excess held-out loss (mnats)')
    axes[0][-1].legend(fontsize=6, loc='upper right', handlelength=1.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def fig_frontier(rows, rng, path):
    plt = style()
    scales = analysis.scales_of(analysis.runs_only(rows))
    fig, axes = plt.subplots(1, len(scales), figsize=(7.1, 2.3), sharey=True, squeeze=False)
    for ax, s in zip(axes[0], scales):
        for i, a in enumerate(arm_rows(rows, s, rng)[0]):
            mean, lo, hi = a['ratio']
            side = 1 if i % 2 else -1  # arms that share a payload sit on one vertical line; alternate the label side
            err = None if lo is None else [[max(0.0, mean - lo)], [max(0.0, hi - mean)]]
            fam = next(f for f, ns in analysis.FAMILIES.items() if a['name'] in ns)
            ax.errorbar(a['payload'], max(mean, 1e-3), yerr=err, fmt='o', ms=3, color=analysis.COLORS[fam], elinewidth=0.6)
            ax.annotate(a['name'], (a['payload'], max(mean, 1e-3)), fontsize=4.5, xytext=(3 * side, 1),
                        textcoords='offset points', family='monospace', ha='left' if side > 0 else 'right')
        ax.axhline(1.0, color='#000000', lw=0.6, ls='--')
        ax.set_yscale('symlog', linthresh=0.1)
        ax.set_title(s, fontsize=8)
        ax.set_xlabel('payload (% of full)')
    axes[0][0].set_ylabel('peak excess loss / $\\varepsilon$')
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def fig_latency(lat_rows, scale, device, path):
    plt = style()
    sel = {r['strategy']: r for r in lat_rows if r['scale'] == scale and r['device'] == device}
    names = [n for n in ARMS if n in sel]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.4), sharey=True)
    palette = ['#2f6fd6', '#e8663a', '#1fa67a', '#8a5cd0', '#d9a400']
    for ax, stages, total, title in ((axes[0], latency.DRAIN, 'drain', 'drain (source)'),
                                     (axes[1], latency.RESTORE, 'restore', 'restore (destination)')):
        left = np.zeros(len(names))
        for k, color in zip(stages, palette):
            w = np.array([sel[n][k] for n in names])
            ax.barh(names, w, left=left, color=color, label=k, height=0.7)
            left += w
        lo = np.array([sel[n][total] - sel[n][f'{total}_p5'] for n in names])
        hi = np.array([sel[n][f'{total}_p95'] - sel[n][total] for n in names])
        ax.errorbar([sel[n][total] for n in names], names, xerr=[lo, hi], fmt='none', ecolor='#000000', elinewidth=0.6,
                    capsize=1.5)
        ax.set_xscale('log')
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('milliseconds')
        ax.legend(fontsize=6, ncol=3, loc='lower right')
    axes[0].invert_yaxis()
    for label in axes[0].get_yticklabels():
        label.set_family('monospace')
        label.set_fontsize(6)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def fig_device(out_dir, dev_rows, path):
    plt = style()
    runs = {}
    for p in Path(out_dir).glob('*/*.json'):
        r = json.loads(p.read_text())
        if r['strategy'] == 'full':
            runs[(r['scale'], r['cond'], r['t0'], r['device'])] = r
    keys = sorted({k[:3] for k in runs if k[1] == 'fp32_nodrop'})
    if not keys:
        return False
    fig, axes = plt.subplots(1, len(keys), figsize=(7.1, 2.0), squeeze=False)
    eps = {(r['scale'], int(r['fork'])): num(r['eps']) for r in dev_rows}
    for ax, (scale, cond, t0) in zip(axes[0], keys):
        ref = runs.get((scale, cond, t0, 'cuda'))
        band = eps.get((scale, t0))
        if band:
            ax.axhspan(-band, band, color='#000000', alpha=0.07, lw=0)
        for device, color in (('cpu', '#2f6fd6'), ('xpu', '#e8663a')):
            r = runs.get((scale, cond, t0, device))
            if r and ref:
                ts = sorted(set(r['val']) & set(ref['val']), key=int)
                ax.plot([int(t) - t0 for t in ts], [1e3 * (r['val'][t] - ref['val'][t]) for t in ts], color=color,
                        label=f'{device} $-$ cuda')
        ax.set_title(f'{scale}, step {t0}', fontsize=8)
        ax.set_xlabel('steps after migration')
    axes[0][0].set_ylabel('held-out loss gap (mnats)')
    axes[0][-1].legend(fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)
    return True


def fig_policy(path):
    plt = style()
    shapes = policy.MODELS['llama-7b']
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    gbps = np.logspace(0, 2, 50)
    for name in ('full', 'w_mq8_vlog8', 'w_vr1', 'wbf16_vr1'):
        nbytes = sum(policy.payload(shapes, name).values())
        ax.plot(gbps, nbytes / (gbps * 1.25e8), color=COLORS[name], label=f'{name} ({nbytes / 1e9:.1f} GB)')
    for s, label in ((30, '30 s notice'), (120, '120 s notice')):
        ax.axhline(s, color='#000000', lw=0.6, ls='--')
        ax.text(1.05, s * 1.08, label, fontsize=6)
    ax.set(xscale='log', yscale='log', xlabel='bandwidth (Gb/s)', ylabel='modelled drain time (s)')
    ax.legend(fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def fig_transient(path, cases=((0.9, 0.99), (0.9, 0.999), (0.9, 0.95))):
    plt = style()
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    k = np.arange(1, 2001)
    for (b1, b2), color in zip(cases, ('#e8663a', '#2f6fd6', '#1fa67a')):
        ax.plot(k, (1 - b1 ** k) / np.sqrt(1 - b2 ** k), color=color, label=f'$\\beta_1$={b1}, $\\beta_2$={b2}')
    ax.axhline(1.0, color='#000000', lw=0.6, ls='--')
    ax.set(xscale='log', xlabel='steps after resume, $k$', ylabel='update / steady-state update')
    ax.legend(fontsize=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def fig_trace(path_csv, path, b1, b2):
    """Measured update size after arrival over the uninterrupted run's, with the closed forms dashed."""
    plt = style()
    rows = analysis.read_csv(path_csv)
    forks = sorted({int(r['fork']) for r in rows})
    fig, axes = plt.subplots(1, len(forks), figsize=(3.45, 2.2), sharey=True, squeeze=False)
    shown = (('w_zero', 'zeros'), ('w_fresh', 'fresh'), ('w_v', 'm_dropped'), ('w_warm', None), ('w_warmv', None))
    for ax, t0 in zip(axes[0], forks):
        for name, rho in shown:
            by = {}
            for r in rows:
                if int(r['fork']) == t0 and r['strategy'] == name:
                    by.setdefault(int(r['k']), []).append(float(r['ratio']))
            if not by:
                continue
            ks = np.array(sorted(by))
            color = COLORS.get(name, '#1f9bb0')
            ax.plot(ks, [np.mean(by[k]) for k in ks], color=color, label=name)
            if rho:
                ax.plot(ks, trace.predicted(rho, b1, b2, ks), color=color, ls='--', lw=0.7)
        ax.axhline(1.0, color='#000000', lw=0.5)
        ax.set(xscale='log', yscale='log', xlabel='steps after arrival, $k$')
        ax.set_yticks([0.5, 1, 2, 4, 8], ['0.5', '1', '2', '4', '8'])
        ax.minorticks_off()
        ax.set_title(f'arrive at step {t0}', fontsize=8)
    axes[0][0].set_ylabel('update size / control')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=5.5, handlelength=1.2, ncol=len(labels), loc='lower center', columnspacing=1.0)
    fig.tight_layout(pad=0.3, rect=(0, 0.07, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return rows


def excess(rho, b1, b2):
    """Movement beyond the uninterrupted run's, in units of one steady update, summed over the whole transient."""
    return float(np.sum(trace.predicted(rho, b1, b2, np.arange(1, 200001)) - 1))


def overshoot(b1, b2):
    k = np.arange(1, 100000)
    u = (1 - b1 ** k) / np.sqrt(1 - b2 ** k)
    return float(u.max()), int(k[u.argmax()])


def chaos_rows(path):
    rep = json.loads(Path(path).read_text())
    lines = []
    for r in rep['rows']:
        lines.append(' & '.join(str(x) if x is not None else '--' for x in (
            r['event'], r['injected_at'], r['world'].replace('->', '$\\to$'), r['steps_redone'], r['payload_MB'], r['drain_s'],
            r['restart_s'], r['init_s'], r['fetch_ms'], r['rebuild_ms'], r['downtime_s'])) + ' \\\\')
    return '\n'.join(lines), rep['summary']


class Macros:
    def __init__(self):
        self.items = {}

    def __call__(self, name, value):
        self.items[name] = value

    def dump(self, path):
        body = ['% generated by mvs.paper; do not edit']
        body += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in self.items.items()]
        Path(path).write_text('\n'.join(body) + '\n')


def build(runs='runs', reports='reports', out='paper/gen', big='gpt-m'):
    runs, reports, out = Path(runs), Path(reports), Path(out)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    rng, mac = np.random.default_rng(0), Macros()
    summary = reports / 'shakespeare_scales' / 'summary.csv'
    rows = analysis.read_csv(summary)
    live = analysis.runs_only(rows)
    seeds = {s: len({r['seed'] for r in live if r['scale'] == s}) for s in analysis.scales_of(live)}
    defaults = load_config(str(ROOT / 'configs' / 'shakespeare_char.yaml'))
    sweep = yaml.safe_load((ROOT / 'configs' / 'experiments' / 'shakespeare_scales.yaml').read_text())
    for key, value in (('warmPasses', defaults.warm_k), ('rewarmSteps', defaults.rewarm_k), ('horizon', sweep['horizon']),
                       ('evalEvery', defaults.eval_every), ('evalSamples', defaults.eval_samples),
                       ('maxSteps', defaults.max_steps), ('warmupSteps', defaults.warmup), ('bootDraws', BOOT)):
        mac(key, value)
    mac('nCells', len(live))
    mac('nArms', len({r['strategy'] for r in live}))
    mac('nSeedsMin', min(seeds.values()))
    mac('nSeedsMax', max(seeds.values()))
    for s, n in seeds.items():
        tag = s.replace('gpt-', '').upper()
        mac(f'seeds{tag}', n)
        mac(f'params{tag}', f"{float(next(r['params'] for r in live if r['scale'] == s)) / 1e6:.1f}")
    eps = [num(r['eps']) for r in rows if r['strategy'] == 'seed_band' and num(r['eps']) is not None]
    mac('epsMin', f'{min(eps):.1f}')
    mac('epsMax', f'{max(eps):.1f}')
    band_lam = [num(r['lam']) for r in rows if r['strategy'] == 'seed_band']
    mac('bandLamMin', f'{min(band_lam):.0f}')
    mac('bandLamMax', f'{max(band_lam):.0f}')

    body, arms = main_table(rows, big, rng)
    (out / 'tab_main.tex').write_text(body + '\n')
    for a in arms:
        tag = ''.join(w.capitalize() for w in a['name'].replace('8', 'Eight').replace('16', 'Sixteen').replace('1', 'One')
                      .replace('4', 'Four').split('_'))
        mac(f'pay{tag}', f"{a['payload']:.0f}")
        mac(f'peak{tag}', f"{a['peak'][0]:.1f}")
        mac(f'ratio{tag}', f"{a['ratio'][0]:.2f}")
        mac(f'lam{tag}', f"{a['lam'][0]:.1f}")
        mac(f'regret{tag}', f"{a['regret'][0]:.2f}")
        mac(f'inband{tag}', f"{a['zero']}/{a['n']}")
        mac(f'cens{tag}', a['cens'])
    body, scales = scale_table(rows, rng)
    (out / 'tab_scale.tex').write_text(body + '\n')
    body, forks = fork_table(rows, big, rng)
    (out / 'tab_fork.tex').write_text(body + '\n')
    mac('forkList', ', '.join(str(f) for f in forks))
    body, worst = bytes_table(rows, big)
    (out / 'tab_bytes.tex').write_text(body + '\n')
    mac('bytesWorstPct', f'{100 * worst:.2f}')
    fig_curves(runs / 'shakespeare_scales', rows, big, out / 'figures' / 'curves.pdf')
    fig_frontier(rows, rng, out / 'figures' / 'frontier.pdf')
    fig_policy(out / 'figures' / 'policy.pdf')
    fig_transient(out / 'figures' / 'transient.pdf')
    for (b1, b2), tag in (((0.9, 0.99), 'A'), ((0.9, 0.999), 'B')):
        peak, at = overshoot(b1, b2)
        mac(f'overshoot{tag}', f'{peak:.2f}')
        mac(f'overshootAt{tag}', at)

    for b2, tag in ((0.99, 'A'), (0.999, 'B'), (0.95, 'C')):
        mac(f'firstNoise{tag}', f"{float(trace.predicted('zeros', 0.9, b2, 1)):.1f}")
        mac(f'excessZeros{tag}', f"{excess('zeros', 0.9, b2):.0f}")
    mac('excessFreshA', f"{excess('fresh', 0.9, 0.99):.0f}")
    mac('deficitM', f"{-excess('m_dropped', 0.9, 0.99):.1f}")
    steps = reports / 'trace' / 'steps.csv'
    if steps.exists():
        trows = fig_trace(steps, out / 'figures' / 'trace.pdf', defaults.beta1, defaults.beta2)
        mac('traceScale', trows[0]['scale'])
        mac('traceSeeds', len({r['seed'] for r in trows}))
        mac('traceSteps', max(int(r['k']) for r in trows))
        for name, tag in (('w_zero', 'Zero'), ('w_fresh', 'Fresh'), ('w_v', 'V'), ('w_warm', 'Warm'), ('w_vr1', 'Rank'),
                          ('w_mq8_vlog8', 'Q')):
            # the late fork is where the model's assumptions hold; the early one is quoted beside it
            tforks = sorted({int(r['fork']) for r in trows})
            for t0, when in ((tforks[-1], ''), (tforks[0], 'Early')):
                for k, word in ((1, 'One'), (10, 'Ten'), (100, 'Hundred')):
                    sel = [float(r['ratio']) for r in trows
                           if r['strategy'] == name and int(r['k']) == k and int(r['fork']) == t0]
                    if sel:
                        mac(f'trace{tag}{word}{when}', f'{np.mean(sel):.2f}')
        for rho, tag in (('zeros', 'Zero'), ('fresh', 'Fresh'), ('m_dropped', 'V')):
            for k, word in ((1, 'One'), (10, 'Ten'), (100, 'Hundred')):
                mac(f'pred{tag}{word}', f'{float(trace.predicted(rho, defaults.beta1, defaults.beta2, k)):.2f}')

    scenarios = [('AWS, 10 Gb/s', dict(notice=120.0, gbps=10.0)), ('AWS, 100 Gb/s', dict(notice=120.0, gbps=100.0)),
                 ('Azure, 10 Gb/s', dict(notice=30.0, gbps=10.0)), ('Azure, 100 Gb/s', dict(notice=30.0, gbps=100.0)),
                 ('GCP, 10 Gb/s', dict(notice=30.0, gbps=10.0)), ('GCP, 50 Gb/s', dict(notice=30.0, gbps=50.0))]
    body, _ = policy_rows(summary, scenarios)
    (out / 'tab_policy.tex').write_text(body + '\n')
    shapes = policy.MODELS['llama-7b']
    mac('llamaParams', f'{sum(math.prod(s) for _, s in shapes) / 1e9:.2f}')
    for name, tag in (('full', 'Full'), ('w_mq8_vlog8', 'Q'), ('w_vr1', 'Rank'), ('wbf16_vr1', 'RankBf')):
        gb = sum(policy.payload(shapes, name).values()) / 1e9
        mac(f'llama{tag}GB', f'{gb:.1f}')
        mac(f'llama{tag}Drain', f'{gb * 8 / 10:.1f}')

    lat = reports / 'timing' / 'latency_cuda.csv'
    if lat.exists():
        body, lat_rows = latency_table(lat, big)
        (out / 'tab_latency.tex').write_text(body + '\n')
        fig_latency(lat_rows, big, 'cuda', out / 'figures' / 'latency.pdf')
        by = {r['strategy']: r for r in lat_rows if r['scale'] == big}
        mac('latReps', by['full']['reps'])
        for name, tag in (('full', 'Full'), ('w_mq8_vlog8', 'Q'), ('w_vr1', 'Rank'), ('w_warm', 'Warm')):
            if name in by:
                mac(f'drain{tag}', f"{by[name]['drain']:.0f}")
                mac(f'restore{tag}', f"{by[name]['restore']:.0f}")
    lat_cpu = reports / 'timing' / 'latency_cpu.csv'
    if lat_cpu.exists():
        (out / 'tab_latency_cpu.tex').write_text(latency_table(lat_cpu, big)[0] + '\n')
    dev = reports / 'xdevice' / 'xdevice.csv'
    if dev.exists():
        body, dev_rows, worst = device_table(dev)
        (out / 'tab_device.tex').write_text(body + '\n')
        fig_device(runs / 'xdevice', dev_rows, out / 'figures' / 'device.pdf')
        for (device, cond), w in worst.items():
            mac(f"devWorst{device.capitalize()}{'Fp' if cond == 'fp32_nodrop' else 'Amp'}", f'{w:.2f}')
    diag = reports / 'shakespeare_scales' / 'diagnostics.csv'
    if diag.exists():
        body, by = diag_table(diag, big)
        (out / 'tab_diag.tex').write_text(body + '\n')
        for name, tag in (('w_zero', 'Zero'), ('w_fresh', 'Fresh'), ('w_warm', 'Warm'), ('w_m', 'M'), ('w_vr1', 'Rank'),
                          ('w_v', 'V')):
            if name in by:
                mac(f'step{tag}', f"{by[name]['step_ratio']:.1f}")
                mac(f'cos{tag}', f"{by[name]['cos']:.2f}")
                mac(f'mnorm{tag}', f"{by[name]['m_ratio']:.1f}")
                mac(f'verr{tag}', f"{100 * by[name]['sqrt_v_err']:.0f}")
    for tag in ('gpu', 'resize'):
        rec = runs / f'chaos_{tag}' / 'recovery.json'
        if rec.exists():
            body, s = chaos_rows(rec)
            (out / f'tab_chaos_{tag}.tex').write_text(body + '\n')
            mac(f'chaos{tag.capitalize()}Goodput', s['goodput'])
            mac(f'chaos{tag.capitalize()}Mttr', s['mttr_s'])
            mac(f'chaos{tag.capitalize()}Redone', s['steps_redone'])
            mac(f'chaos{tag.capitalize()}Gap', f"{1e3 * abs(s['final_val_gap']):.1f}")
    mac.dump(out / 'numbers.tex')
    return mac.items
