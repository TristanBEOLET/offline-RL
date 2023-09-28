import torch


def AvgL1Norm(x: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """Average L1 norm function.

    :param x: input tensor
    :param eps: min value, defaults to 1e-8
    :return: output tensor
    """
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)
