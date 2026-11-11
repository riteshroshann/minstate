"""A migration between two processes over a paced TCP link.

The source drains a payload through a socket limited to a chosen bandwidth; the destination, a separate process that
may sit on another device, restores it and trains on. The link is emulated: pacing reproduces bandwidth, not loss,
jitter or congestion.
"""
import json
import socket
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import analysis, manifest
from .config import load
from .experiment import forks
from .state import payload
from .trainer import Trainer

CHUNK = 1 << 16


def spec(path):
    with open(path) as f:
        e = yaml.safe_load(f)
    e.setdefault('seed', 1337)
    e.setdefault('reps', 3)
    e.setdefault('steps', 20)
    e.setdefault('port', 47613)
    return e


def config(e, scale, side):
    merged = {**e['scales'][scale]['model'], **e[side].get('overrides', {}), 'seed': e['seed'], 'device': e[side]['device']}
    return load(e['base'], [f'{k}={json.dumps(v)}' for k, v in merged.items()])


def send_json(sock, obj):
    body = json.dumps(obj).encode()
    sock.sendall(struct.pack('<I', len(body)) + body)


def recv_exact(sock, buf):
    view, got = memoryview(buf), 0
    while got < len(view):
        n = sock.recv_into(view[got:], min(CHUNK, len(view) - got))
        if not n:
            raise ConnectionError('link closed mid-payload')
        got += n
    return buf


def recv_json(sock):
    n = struct.unpack('<I', bytes(recv_exact(sock, bytearray(4))))[0]
    return json.loads(bytes(recv_exact(sock, bytearray(n))))


def send_paced(sock, buf, mbps):
    """Send buf no faster than mbps megabits per second. mbps None is the unthrottled loopback, the floor."""
    view, sent, t0 = memoryview(buf), 0, time.perf_counter()
    rate = mbps * 125000 if mbps else None
    while sent < len(view):
        sent += sock.send(view[sent:sent + CHUNK])
        if rate:
            # pace against the cumulative budget, so one late chunk does not slow the whole transfer
            ahead = sent / rate - (time.perf_counter() - t0)
            if ahead > 0:
                time.sleep(ahead)


def serve(path, port):
    """Destination: take payloads until told to stop. Each is acknowledged on its last byte, then restored and resumed."""
    e, scale, tr = spec(path), None, None
    with socket.create_server(('127.0.0.1', port)) as srv:
        conn, _ = srv.accept()
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while True:
                head = recv_json(conn)
                if head.get('bye'):
                    return
                buf = recv_exact(conn, bytearray(head['nbytes']))
                conn.sendall(b'A')  # the drain ends here: every byte has left the source
                if head['scale'] != scale:
                    tr = None  # release the previous model before building the next
                    scale, tr = head['scale'], Trainer(config(e, head['scale'], 'dest'))
                clock, t0 = payload.Clock(tr.device), time.perf_counter()
                meta, passes, warn = payload.unpack(tr, buf, clock)
                restore = time.perf_counter() - t0
                # training on is slow on a small device, so it is asked for once per payload and not per repetition
                val, after, ms = None, None, []
                if head['steps']:
                    val = tr.evaluate()
                    for _ in range(head['steps']):
                        c = time.perf_counter()
                        tr.step()
                        ms.append((time.perf_counter() - c) * 1e3)
                    after = tr.evaluate()
                send_json(conn, {'restore_s': restore, 'restore_ms': clock.ms, 'val_arrival': val, 'step': meta['step'],
                                 'val_after': after, 'step_ms': statistics.median(ms) if ms else None,
                                 'passes': passes, 'env': manifest.environment(tr.device)})


def connect(port, proc, timeout=180):
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError(f'destination exited with status {proc.returncode}')
        try:
            return socket.create_connection(('127.0.0.1', port), timeout=5)
        except OSError:
            time.sleep(0.5)
    raise TimeoutError('destination did not start listening')


def run(path):
    """Source: start the destination process, then drain every (scale, bandwidth, strategy) cell through the link."""
    e, rows, dest_env = spec(path), [], None
    out, report = Path(e['out']), Path(e['report'])
    out.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    script = manifest.ROOT / 'scripts' / 'link.py'
    # the destination may live in another environment, e.g. one built for a different accelerator
    python = str(manifest.ROOT / e['dest']['python']) if e['dest'].get('python') else sys.executable
    with open(out / 'dest.log', 'w') as log:
        proc = subprocess.Popen([python, '-u', str(script), 'serve', path],
                                cwd=manifest.ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            sock = connect(e['port'], proc)
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            for scale, s in e['scales'].items():
                cfg = config(e, scale, 'source')
                src = Trainer(cfg)
                t0 = forks(cfg, [e['fork']])[0]
                payload.load(src, str(Path(s['snaps']) / scale / f"s{e['seed']}" / 'control' / f'snap_{t0}.mvs'))
                val_src = src.evaluate()
                for mbps in e['bandwidths_mbps']:
                    for name in e['strategies']:
                        for rep in range(e['reps']):
                            clock, start = payload.Clock(src.device), time.perf_counter()
                            buf, _ = payload.pack(src, name, clock)
                            check = rep == 0 and mbps == e['bandwidths_mbps'][0]
                            send_json(sock, {'scale': scale, 'nbytes': len(buf), 'steps': e['steps'] if check else 0})
                            send_paced(sock, buf, mbps)
                            recv_exact(sock, bytearray(1))
                            drain = time.perf_counter() - start
                            res = recv_json(sock)
                            dest_env = res.pop('env')
                            model = len(buf) * 8 / (mbps * 1e6) if mbps else None
                            rows.append({'scale': scale, 'params': src.n_params(), 'fork': t0, 'strategy': name,
                                         'mbps': mbps or 0, 'rep': rep, 'bytes': len(buf),
                                         'pack_s': sum(clock.ms.values()) / 1e3, 'drain_s': drain, 'model_s': model,
                                         'notice_s': e['notice'], 'fits': int(drain <= e['notice']),
                                         'restore_s': res['restore_s'], 'val_source': val_src,
                                         'val_arrival': res['val_arrival'], 'val_after': res['val_after'],
                                         'dest_step_ms': res['step_ms'], 'passes': res['passes']})
                            print(f"{scale} {mbps or 'loopback':>8} Mb/s {name:12s} {len(buf) / 1e6:8.1f} MB  drain "
                                  f"{drain:7.2f} s  model {model or 0:7.2f} s  restore {res['restore_s']:5.2f} s"
                                  + (f"  val {val_src:.4f} -> {res['val_arrival']:.4f}" if check else ''), flush=True)
                            analysis.write_csv(rows, report / 'link.csv')
                del src
            send_json(sock, {'bye': True})
            sock.close()
            proc.wait(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()
    (report / 'link.json').write_text(json.dumps({'source': manifest.environment(e['source']['device']),
                                                   'destination': dest_env, 'experiment': e}, indent=1))
    return rows
