

import torch
import torch.nn as nn


def clip_grad(model: nn.Module, max_norm: float) -> None:

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
