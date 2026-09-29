import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRA(nn.Module):
    def __init__(self, base, r, alpha, p):
        super().__init__()
        dev = next(base.parameters()).device
        self.base, self.scale, self.drop = base, alpha / r, nn.Dropout(p)
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        z = F.linear(F.linear(self.drop(x).to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return self.base(x) + (z * self.scale).to(x.dtype)


class QLoRA(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x, y):
        logits = self.base(input_ids=x).logits
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))

    def mvs_buffers(self):
        return {}


def load_base(cfg, vocab, device):
    import transformers as tf
    if cfg.hf_model == 'tiny':
        conf = tf.LlamaConfig(vocab_size=vocab, hidden_size=64, intermediate_size=176, num_hidden_layers=2,
                              num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=cfg.block_size)
        return tf.LlamaForCausalLM(conf).to(device)
    quant = None if cfg.quant == 'none' else tf.BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type=cfg.quant, bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
    where = device.index if device.type == 'cuda' else 'cpu'
    return tf.AutoModelForCausalLM.from_pretrained(cfg.hf_model, revision=cfg.hf_revision, quantization_config=quant,
                                                   torch_dtype=torch.bfloat16, device_map={'': where})


def build(cfg, vocab, device):
    base = load_base(cfg, vocab, device)
    base.requires_grad_(False)
    base.config.use_cache = False
    torch.manual_seed(cfg.seed + 1)
    for mod in list(base.modules()):
        for name in cfg.lora_targets:
            child = getattr(mod, name, None)
            if isinstance(child, nn.Module) and hasattr(child, 'in_features'):
                setattr(mod, name, LoRA(child, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout))
    if cfg.grad_ckpt and cfg.hf_model != 'tiny':
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        base.enable_input_require_grads()
    return QLoRA(base)
