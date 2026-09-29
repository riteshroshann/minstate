import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c_attn = nn.Linear(c.n_embd, 3 * c.n_embd, bias=c.bias)
        self.c_proj = nn.Linear(c.n_embd, c.n_embd, bias=c.bias)
        self.heads, self.p = c.n_head, c.dropout
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = (z.view(b, t, self.heads, c // self.heads).transpose(1, 2) for z in self.c_attn(x).split(c, 2))
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p if self.training else 0.0, is_causal=True)
        return self.drop(self.c_proj(y.transpose(1, 2).reshape(b, t, c)))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln_1 = nn.LayerNorm(c.n_embd, bias=c.bias)
        self.attn = Attention(c)
        self.ln_2 = nn.LayerNorm(c.n_embd, bias=c.bias)
        self.mlp = nn.Sequential(nn.Linear(c.n_embd, 4 * c.n_embd, bias=c.bias), nn.GELU(),
                                 nn.Linear(4 * c.n_embd, c.n_embd, bias=c.bias), nn.Dropout(c.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, c, vocab):
        super().__init__()
        self.wte = nn.Embedding(vocab, c.n_embd)
        self.wpe = nn.Embedding(c.block_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.h = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd, bias=c.bias)
        self.lm_head = nn.Linear(c.n_embd, vocab, bias=False)
        self.lm_head.weight = self.wte.weight
        for n, p in self.named_parameters():
            if p.dim() > 1:
                deep = n.endswith('c_proj.weight') or n.endswith('mlp.2.weight')
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer) if deep else 0.02)
            elif n.endswith('bias'):
                nn.init.zeros_(p)

    def forward(self, idx, y):
        x = self.drop(self.wte(idx) + self.wpe(torch.arange(idx.shape[1], device=idx.device)))
        for block in self.h:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
