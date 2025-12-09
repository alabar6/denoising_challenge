import torch


def jahne_noise(input: torch.Tensor, gain: float = 0.0, sigma: float = 0.0):
    """
    Modeling of Jahne noise.
    """
    normal = 0
    if sigma > 0:
        normal = torch.normal(torch.zeros(input.shape), sigma).to(input.device)
    if gain <= 0:
        return input + normal
    thr = torch.tensor(1e-12)
    return torch.poisson(torch.maximum(input / gain, thr)) * gain + normal