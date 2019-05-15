import torch


def cross_entropy(prediction, truth):
    """
    Returns the multi-class cross-entropy between prediction and groudn truh. Assumes
    that the input arguments are of the same size.

    """
    eps = 1e-10
    return torch.sum(-1 * truth * torch.log(prediction + eps))


def dice(prediction, truth):
    """
    Returns the negative dice coefficient between prediction and grounf truth.

    """
    numerator = -2 * torch.sum(prediction * truth)
    denominator = torch.sum(prediction**2 + truth**2)
    return numerator / denominator
