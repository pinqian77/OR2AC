import torch
import torch.nn.functional as F

def quantile_regression_loss(input, target, tau, weight):
    """
    input: (N, T)
    target: (N, T)
    tau: (N, T)
    """
    input = input.unsqueeze(-1)
    target = target.detach().unsqueeze(-2)
    tau = tau.detach().unsqueeze(-1)
    weight = weight.detach().unsqueeze(-2)
    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    L = F.smooth_l1_loss(expanded_input, expanded_target, reduction="none")  # (N, T, T)
    sign = torch.sign(expanded_input - expanded_target) / 2. + 0.5
    rho = torch.abs(tau - sign) * L * weight
    return rho.sum(dim=-1).mean()

def distortion_fn(tau, 
                  mode = "neutral", 
                  param = 0.):
    # Risk distortion function
    tau = tau.clamp(0., 1.)
    if param >= 0:
        if mode == "neutral":
            tau_ = tau
        elif mode == "cvar":
            tau_ = (1. / param) * tau
        return tau_.clamp(0., 1.)
    else:
        return 1 - distortion_fn(1 - tau, mode, -param)


def distortion_de(tau, 
                  mode = "neutral", 
                  param = 0., 
                  eps = 1e-8):
    # Derivative of Risk distortion function
    tau = tau.clamp(0., 1.)
    if param >= 0:
        if mode == "neutral":
            tau_ = torch.ones_like(tau)
            # tau_ = ptu.ones_like(tau)
        elif mode == "cvar":
            tau_ = (1. / param) * (tau < param)
        return tau_.clamp(0., 5.)
    else:
        return distortion_de(1 - tau, mode, -param)
