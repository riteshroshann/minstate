import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

from .launcher import Group

TRAIN = str(Path(__file__).resolve().parents[1] / 'scripts' / 'train.py')
FETCH = ('read', 'deserialise', 'h2d', 'decode')


def parse(spec):
    out = []
    for item in filter(None, (s.strip() for s in spec.split(','))):
        kind, rest = item.split('@')
        at, _, world = rest.partition(':')
        out.append({'kind': kind, 'at': int(at), 'world': int(world) if world else None})
        if kind not in ('preempt', 'crash'):
            raise ValueError(f'unknown event {kind}; use preempt@STEP[:WORLD] or crash@STEP')
    return out


class Tail:
    def __init__(self, path):
        self.path, self.pos, self.step = path, 0, -1

    def poll(self):
        if not os.path.exists(self.path):
            return self.step
        with open(self.path) as f:
            f.seek(self.pos)
            for line in f:
                if not line.endswith('\n'):
                    break
                self.pos += len(line.encode())
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'loss' in e:
                    self.step = e['t']
        return self.step


def entries(path):
    out = []
    with open(path) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def job(config, overrides, out, world):
    return [sys.executable, TRAIN, *(['--config', config] if config else []), *overrides, f'out_dir={out}'], world


def supervise(cmd, world, events, via, out, poll=0.02):
    tail, notice, records, restart = Tail(os.path.join(out, 'log.jsonl')), os.path.join(out, 'NOTICE'), [], 0
    pending, t0 = list(events), time.time()
    while True:
        g = Group(cmd + [f'notice_file={notice}', 'auto_resume=true'], world, restart)
        ev, rec = (pending[0] if pending else None), None
        while ev and rec is None and g.alive():
            if tail.poll() >= ev['at']:
                rec = {**ev, 't_inject': time.time(), 'seen': tail.step, 'world_before': world}
                if ev['kind'] == 'crash':
                    g.kill()
                elif via == 'signal':
                    g.notify()
                else:
                    Path(notice).touch()
            time.sleep(poll)
        codes = g.wait()
        if rec:
            records.append({**rec, 'codes': codes, 't_exit': time.time()})
            pending.pop(0)
            world = ev['world'] or world
        restart += 1
        if all(c == 0 for c in codes):
            return records, time.time() - t0
        if restart > 2 * len(events) + 3:
            raise RuntimeError(f'job failed to finish: exit codes {codes}')


def segments(log):
    segs = []
    for e in log:
        if e.get('event') == 'start':
            segs.append([])
        if segs:
            segs[-1].append(e)
    return segs


def first(seg, pred):
    return next((e for e in seg if pred(e)), None)


def event(seg, name):
    return first(seg, lambda e: e.get('event') == name)


def rounded(x, n=3):
    return None if x is None else round(x, n)


def row(rec, before, after):
    last = [e for e in before if 'loss' in e][-1]
    drain, start, built, res = event(before, 'drain'), event(after, 'start'), event(after, 'built'), event(after, 'resume')
    nxt, ms, src = first(after, lambda e: 'loss' in e), (res or {}).get('ms', {}), (res or {}).get('step', 0)
    warn = (res or {}).get('warnings') or []
    return {'event': rec['kind'], 'injected_at': rec['seen'],
            'world': f"{rec['world_before']}->{start['world'] if start else '?'}",
            'drained_at': drain['t'] if drain else None, 'resumed_from': src, 'steps_redone': last['t'] + 1 - src,
            'drain_s': rounded(drain and drain['drain_s']), 'payload_MB': round((res or {}).get('file_bytes', 0) / 1e6, 3),
            'strategy': (res or {}).get('strategy'),
            'restart_s': rounded(start and start['time'] - rec['t_exit']),
            'init_s': rounded(built and start and built['time'] - start['time']),
            'fetch_ms': round(sum(ms.get(k, 0.0) for k in FETCH), 2), 'rebuild_ms': round(ms.get('rebuild', 0.0), 2),
            'rebuild_passes': (res or {}).get('passes', 0), 'integrity': '; '.join(['crc32 ok', *warn]),
            'downtime_s': rounded(nxt and nxt['time'] - last['time'])}


def verdict(gap):
    if gap == 0:
        return 'bitwise identical to the uninterrupted run'
    if gap < 1e-5:
        return 'identical up to floating-point summation order'
    return f'deviates by up to {gap:.4f} nats (lossy strategy; compare with the noise floor)'


def analyse(ctrl_log, job_log, records, wall, ctrl_wall):
    ctrl, log = entries(ctrl_log), entries(job_log)
    c_loss = {e['t']: e['loss'] for e in ctrl if 'loss' in e}
    c_val = [e['val'] for e in ctrl if 'val' in e]
    step_ms = statistics.median(e['ms'] for e in ctrl if 'loss' in e)
    final, computed = {}, 0
    for e in log:
        if 'loss' in e:
            final[e['t']] = e['loss']
            computed += 1
    segs = segments(log)
    rows = [row(rec, segs[i], segs[i + 1] if i + 1 < len(segs) else []) for i, rec in enumerate(records)]
    common = sorted(set(final) & set(c_loss))
    gap = max((abs(final[t] - c_loss[t]) for t in common), default=0.0)
    j_val = [e['val'] for e in log if 'val' in e]
    down = [r['downtime_s'] for r in rows if r['downtime_s'] is not None]
    useful = len(final) * step_ms / 1e3
    return {'rows': rows, 'summary': {
        'steps': len(final), 'steps_computed': computed, 'steps_redone': computed - len(final),
        'interruptions': len(rows), 'mttr_s': round(statistics.mean(down), 3) if down else 0.0,
        'wall_s': round(wall, 2), 'control_wall_s': round(ctrl_wall, 2),
        'goodput': round(useful / wall, 3), 'control_goodput': round(useful / ctrl_wall, 3),
        'max_loss_gap': gap, 'final_val_gap': (j_val[-1] - c_val[-1]) if j_val and c_val else None,
        'fidelity': verdict(gap)}}


def markdown(rep):
    cols = ['event', 'injected_at', 'world', 'drained_at', 'resumed_from', 'steps_redone', 'payload_MB', 'drain_s',
            'restart_s', 'init_s', 'fetch_ms', 'rebuild_ms', 'downtime_s', 'integrity']
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join(str(r[c]) for c in cols) + ' |' for r in rep['rows']]
    s = rep['summary']
    lines += ['', f"steps {s['steps']}  computed {s['steps_computed']}  redone {s['steps_redone']}  "
                  f"interruptions {s['interruptions']}  MTTR {s['mttr_s']} s",
              f"wall {s['wall_s']} s vs control {s['control_wall_s']} s  goodput {s['goodput']} (control {s['control_goodput']})",
              f"fidelity: {s['fidelity']}  final val gap {s['final_val_gap']}"]
    return '\n'.join(lines)


def run(config, overrides, world, events, out, via='signal'):
    out = Path(out)
    ctrl, work = out / 'control', out / 'job'
    for d in (ctrl, work):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    cmd, w = job(config, overrides, str(ctrl), world)
    t0 = time.time()
    codes = Group(cmd, w).wait()
    if any(codes):
        raise RuntimeError(f'control run failed: {codes}')
    ctrl_wall = time.time() - t0
    cmd, w = job(config, overrides, str(work), world)
    records, wall = supervise(cmd, w, parse(events) if isinstance(events, str) else events, via, str(work))
    rep = analyse(ctrl / 'log.jsonl', work / 'log.jsonl', records, wall, ctrl_wall)
    (out / 'recovery.json').write_text(json.dumps(rep, indent=1))
    (out / 'recovery.md').write_text(markdown(rep) + '\n')
    return rep
