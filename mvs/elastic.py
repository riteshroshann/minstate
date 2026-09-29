import argparse
import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request

import torch
import torch.distributed as dist

from .config import dump, load
from .state import payload, wire
from .trainer import Trainer

DRAINED = 3


def aws():
    put = urllib.request.Request('http://169.254.169.254/latest/api/token', method='PUT',
                                 headers={'X-aws-ec2-metadata-token-ttl-seconds': '60'})
    token = urllib.request.urlopen(put, timeout=1).read().decode()
    get = urllib.request.Request('http://169.254.169.254/latest/meta-data/spot/instance-action',
                                 headers={'X-aws-ec2-metadata-token': token})
    try:
        urllib.request.urlopen(get, timeout=1)
        return True
    except urllib.error.HTTPError:
        return False


def gcp():
    req = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/preempted',
                                 headers={'Metadata-Flavor': 'Google'})
    return urllib.request.urlopen(req, timeout=1).read().decode().strip() == 'TRUE'


class Notice:
    def __init__(self, cfg):
        self.at, self.source, self.file = None, None, cfg.notice_file
        for name in ('SIGTERM', 'SIGBREAK'):
            if hasattr(signal, name):
                try:
                    signal.signal(getattr(signal, name), self.fire)
                except ValueError:
                    pass
        for on, probe in ((cfg.aws_spot, aws), (cfg.gcp_spot, gcp)):
            if on:
                threading.Thread(target=self.watch, args=(probe,), daemon=True).start()

    def fire(self, *_, source='signal'):
        if self.at is None:
            self.at, self.source = time.time(), source

    def watch(self, probe):
        while self.at is None:
            try:
                if probe():
                    self.fire(source=probe.__name__)
            except OSError:
                pass
            time.sleep(5)

    def local(self):
        if self.at is None and self.file and os.path.exists(self.file):
            self.fire(source='file')
        return self.at is not None

    def agreed(self, tr):
        flag = torch.tensor([1.0 if self.local() else 0.0], device=tr.device)
        if tr.world > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return flag.item() > 0


def torn(path):
    if not os.path.exists(path) or not os.path.getsize(path):
        return False
    with open(path, 'rb') as f:
        f.seek(-1, 2)
        return f.read(1) != b'\n'


class Log:
    def __init__(self, path, on=True):
        self.f = None
        if on:
            broken = torn(path)
            self.f = open(path, 'a', buffering=1)
            if broken:
                self.f.write('\n')

    def __call__(self, **kw):
        if self.f:
            self.f.write(json.dumps({'time': time.time(), **kw}) + '\n')


def env():
    return tuple(int(os.environ.get(k, 0 if k != 'WORLD_SIZE' else 1)) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))


def latest(out):
    found = []
    for name in ('payload.mvs', 'ckpt.mvs'):
        path = os.path.join(out, name)
        try:
            meta = wire.peek(path)
            found.append((meta['step'], meta['time'], path))
        except (OSError, ValueError, KeyError):
            pass
    return max(found)[2] if found else ''


def finish(world):
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def crash():
    if hasattr(signal, 'SIGKILL'):
        os.kill(os.getpid(), signal.SIGKILL)
    os._exit(137)


def drain(tr, cfg, notice, log):
    if tr.rank == 0:
        at = notice.at or time.time()
        info = payload.save(tr, cfg.strategy, os.path.join(cfg.out_dir, 'payload.mvs'))
        log(event='drain', t=tr.t, source=notice.source, notice_at=at, drain_s=time.time() - at, **info)
        if notice.source == 'file' and os.path.exists(cfg.notice_file):
            os.remove(cfg.notice_file)
    finish(tr.world)
    sys.exit(DRAINED)


def run(cfg):
    rank, world, local = env()
    restart = int(os.environ.get('TORCHELASTIC_RESTART_COUNT', os.environ.get('MVS_RESTART', 0)))
    os.makedirs(cfg.out_dir, exist_ok=True)
    log = Log(os.path.join(cfg.out_dir, 'log.jsonl'), rank == 0)
    log(event='start', world=world, restart=restart, pid=os.getpid())
    if world > 1:
        nccl = dist.is_nccl_available() and torch.cuda.is_available() and cfg.device != 'cpu'
        dist.init_process_group('nccl' if nccl else 'gloo')
    notice = Notice(cfg)
    tr = Trainer(cfg, rank, world, local)
    log(event='built', world=world, accum=tr.accum, params=tr.n_params(), device=str(tr.device))
    if rank == 0:
        dump(cfg, os.path.join(cfg.out_dir, 'config.yaml'))
    src = cfg.resume or (latest(cfg.out_dir) if cfg.auto_resume else '')
    if src:
        log(event='resume', src=src, **payload.load(tr, src))
    snaps = {int(round(s * cfg.max_steps)) if isinstance(s, float) else int(s) for s in cfg.snapshots}
    while tr.t < cfg.max_steps:
        if notice.agreed(tr):
            drain(tr, cfg, notice, log)
        if tr.t == cfg.crash_at and restart == 0 and rank == world - 1:
            crash()
        t, t0 = tr.t, time.perf_counter()
        loss = tr.step()
        log(t=t, loss=loss, lr=tr.lr_at(t), ms=(time.perf_counter() - t0) * 1e3)
        if rank == 0 and cfg.eval_every and tr.t % cfg.eval_every == 0:
            log(t=tr.t, val=tr.evaluate())
        if rank == 0 and tr.t in snaps:
            payload.save(tr, 'full', os.path.join(cfg.out_dir, f'snap_{tr.t}.mvs'))
        if rank == 0 and cfg.ckpt_every and tr.t % cfg.ckpt_every == 0:
            log(event='ckpt', **payload.save(tr, cfg.ckpt_strategy, os.path.join(cfg.out_dir, 'ckpt.mvs')))
    if rank == 0 and cfg.save_final:
        payload.save(tr, 'full', os.path.join(cfg.out_dir, 'final.mvs'))
    log(event='done', t=tr.t)
    finish(world)
    return 0


def cli(argv=None):
    ap = argparse.ArgumentParser(description='train with preemption-aware elastic migration')
    ap.add_argument('--config')
    for flag in ('--strategy', '--resume', '--ckpt-strategy', '--notice-file', '--out-dir'):
        ap.add_argument(flag)
    ap.add_argument('--ckpt-every', type=int)
    for flag in ('--aws-spot', '--gcp-spot', '--auto-resume'):
        ap.add_argument(flag, action='store_true')
    ap.add_argument('overrides', nargs='*', help='key=value')
    a = ap.parse_args(argv)
    extra = [f'{k}={v}' for k, v in (('strategy', a.strategy), ('resume', a.resume), ('ckpt_strategy', a.ckpt_strategy),
                                     ('notice_file', a.notice_file), ('out_dir', a.out_dir), ('ckpt_every', a.ckpt_every))
             if v is not None]
    extra += [f'{k}=true' for k in ('aws_spot', 'gcp_spot', 'auto_resume') if getattr(a, k)]
    return run(load(a.config, [*a.overrides, *extra]))
