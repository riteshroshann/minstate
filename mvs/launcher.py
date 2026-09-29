import os
import signal
import socket
import subprocess
import time

DRAINED = 3


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class Group:
    def __init__(self, cmd, nproc, restart=0, env=None):
        port, flags = free_port(), getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0)
        base = {**os.environ, **({'USE_LIBUV': '0'} if os.name == 'nt' else {}), **(env or {})}
        self.procs = []
        for r in range(nproc):
            e = {**base, 'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port), 'RANK': str(r), 'WORLD_SIZE': str(nproc),
                 'LOCAL_RANK': str(r), 'LOCAL_WORLD_SIZE': str(nproc), 'MVS_RESTART': str(restart)}
            self.procs.append(subprocess.Popen(cmd, env=e, creationflags=flags))

    def alive(self):
        return any(p.poll() is None for p in self.procs)

    def notify(self):
        sig = getattr(signal, 'CTRL_BREAK_EVENT', signal.SIGTERM)
        for p in self.procs:
            if p.poll() is None:
                p.send_signal(sig)

    def kill(self, rank=-1):
        self.procs[rank].kill()

    def wait(self, poll=0.05):
        while True:
            codes = [p.poll() for p in self.procs]
            if None not in codes:
                return codes
            if any(c not in (None, 0, DRAINED) for c in codes):
                for p in self.procs:
                    if p.poll() is None:
                        p.kill()
                return [p.wait() for p in self.procs]
            time.sleep(poll)


def launch(cmd, nproc=1, max_restarts=0, env=None):
    current = []

    def forward(*_):
        for g in current:
            g.notify()

    for name in ('SIGTERM', 'SIGBREAK'):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), forward)
    for restart in range(max_restarts + 1):
        current[:] = [Group(cmd, nproc, restart, env)]
        codes = current[0].wait()
        if all(c == 0 for c in codes):
            return 0
        if all(c == DRAINED for c in codes):
            return DRAINED
    return 1
