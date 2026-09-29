import torch


def build(cfg, vocab, device):
    torch.manual_seed(cfg.seed)
    if cfg.model == 'gpt':
        from .gpt import GPT
        return GPT(cfg, vocab).to(device)
    if cfg.model == 'resnet18':
        from .resnet import ResNet18
        return ResNet18().to(device, memory_format=torch.channels_last)
    if cfg.model == 'qlora':
        from .qlora import build as qlora
        return qlora(cfg, vocab, device)
    raise ValueError(f'unknown model {cfg.model}')
