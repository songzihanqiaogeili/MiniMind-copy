"""单机预训练工具。"""

import math
import random

import torch


def setup_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr(step: int, total_steps: int, peak_lr: float) -> float:
    """按优化器更新次数进行余弦衰减，最低为初始学习率的 10%。"""
    progress = step / max(total_steps - 1, 1)
    return peak_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
