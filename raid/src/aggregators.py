import torch

# Probabilities are clamped away from 0 and 1 so the log-odds below stay finite.
_EPS = 1e-7


def all_or_nothing(probabilities: torch.Tensor) -> float:
    """Combine per-paragraph probabilities as independent evidence.

    Mathematically P = prod(p) / (prod(p) + prod(1 - p)). Computing the products
    directly underflows to 0/0 = NaN for long texts, so the same quantity is
    computed in log space: P = sigmoid(sum(log p) - sum(log(1 - p))).
    """
    p = probabilities.double().clamp(_EPS, 1 - _EPS)
    log_odds = torch.log(p).sum() - torch.log1p(-p).sum()
    P = torch.sigmoid(log_odds)

    return P.item()

def take_max(probabilities: torch.Tensor) -> float:
    return probabilities.max().item()

def take_min(probabilities: torch.Tensor) -> float:
    return probabilities.min().item()

def take_mean(probabilities: torch.Tensor) -> float:
    return probabilities.mean().item()
