import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

M64 = (1 << 64) - 1
TRAIN, RECON, EVAL, NOISE = 0, 1, 2, 3
URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


def splitmix(x):
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def mix(*xs):
    h = 0
    for x in xs:
        h = splitmix(h ^ (int(x) & M64))
    return h >> 1


def gen(device, *xs):
    g = torch.Generator(device=device)
    g.manual_seed(mix(*xs))
    return g


def text(cfg):
    p = Path(cfg.data_dir) / 'shakespeare_char' / 'input.txt'
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, p)
    return p.read_text(encoding='utf-8')


class Text:
    def __init__(self, cfg, device, tokens, vocab):
        n = int(0.9 * len(tokens))
        self.cfg, self.device, self.vocab = cfg, device, vocab
        self.train, self.val = tokens[:n].to(device), tokens[n:].to(device)
        self.off = torch.arange(cfg.block_size + 1, device=device)
        self.eval_set = self.draw(self.val, cfg.eval_samples, gen(device, cfg.seed, EVAL))

    def draw(self, data, n, g):
        ix = torch.randint(len(data) - self.cfg.block_size - 1, (n,), generator=g, device=self.device)
        w = data[ix[:, None] + self.off].long()
        return w[:, :-1], w[:, 1:]

    def batch(self, t, g, stream, sub=0):
        return self.draw(self.train, self.cfg.batch_size, gen(self.device, self.cfg.seed, stream, t, g, sub))

    def evals(self):
        x, y, b = *self.eval_set, self.cfg.eval_batch
        return [(x[i:i + b], y[i:i + b]) for i in range(0, len(x), b)]


class Images:
    MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

    def __init__(self, cfg, device):
        self.cfg, self.device, self.vocab = cfg, device, 10
        if cfg.synthetic:
            g = torch.Generator().manual_seed(cfg.seed)
            xtr = torch.randint(0, 256, (cfg.synthetic, 32, 32, 3), generator=g, dtype=torch.uint8)
            ytr = torch.randint(0, 10, (cfg.synthetic,), generator=g)
            xte, yte = xtr, ytr
        else:
            import torchvision
            root = Path(cfg.data_dir) / 'cifar10'
            tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
            te = torchvision.datasets.CIFAR10(root, train=False, download=True)
            xtr, ytr = torch.from_numpy(tr.data), torch.tensor(tr.targets)
            xte, yte = torch.from_numpy(te.data), torch.tensor(te.targets)
        self.x = F.pad(self.norm(xtr), (4, 4, 4, 4)).to(device)
        self.y, self.n = ytr.to(device), len(ytr)
        k = cfg.eval_samples
        self.ex = self.norm(xte[:k]).to(device).contiguous(memory_format=torch.channels_last)
        self.ey = yte[:k].to(device)
        self.ar = torch.arange(32, device=device)

    def norm(self, x):
        m, s = torch.tensor(self.MEAN), torch.tensor(self.STD)
        return ((x.float() / 255 - m) / s).permute(0, 3, 1, 2).contiguous()

    def batch(self, t, g, stream, sub=0):
        c = self.cfg
        k = ((t * c.micro_batches + g) * c.batch_size + torch.arange(c.batch_size)) if not sub else \
            torch.randint(self.n, (c.batch_size,), generator=gen('cpu', c.seed, stream, t, g, sub))
        e, idx = k // self.n, torch.empty(c.batch_size, dtype=torch.long)
        for ep in e.unique().tolist():
            perm = torch.randperm(self.n, generator=gen('cpu', c.seed, stream, ep))
            sel = e == ep
            idx[sel] = perm[k[sel] % self.n]
        idx = idx.to(self.device)
        gg = gen(self.device, c.seed, stream, t, g, sub, 1)
        oy, ox = torch.randint(0, 9, (2, c.batch_size), generator=gg, device=self.device)
        flip = torch.rand(c.batch_size, generator=gg, device=self.device) < 0.5
        r, q = (oy[:, None] + self.ar)[:, :, None], (ox[:, None] + self.ar)[:, None, :]
        x = self.x[idx[:, None, None], :, r, q].permute(0, 3, 1, 2)
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        return x.contiguous(memory_format=torch.channels_last), self.y[idx]

    def evals(self):
        b = self.cfg.eval_batch
        return [(self.ex[i:i + b], self.ey[i:i + b]) for i in range(0, len(self.ex), b)]


def chars(cfg, device):
    s = text(cfg)
    vocab = sorted(set(s))
    idx = {ch: i for i, ch in enumerate(vocab)}
    return Text(cfg, device, torch.tensor([idx[ch] for ch in s], dtype=torch.int16), len(vocab))


def build(cfg, device):
    if cfg.workload == 'shakespeare_char':
        return chars(cfg, device)
    if cfg.workload == 'shakespeare_hf':
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.hf_model, revision=cfg.hf_revision)
        ids = torch.tensor(tok(text(cfg))['input_ids'], dtype=torch.int32)
        return Text(cfg, device, ids, len(tok))
    if cfg.workload == 'cifar10':
        return Images(cfg, device)
    raise ValueError(f'unknown workload {cfg.workload}')
