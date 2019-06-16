import torch
from utils import encode_images
from dataset import SegmentationDataset, transform_test


def evaluate(model, datapath, split, device):
    dataset = SegmentationDataset(root=datapath, year='2009', image_set=split,
                                  transform=transform_test)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    loss = torch.zeros(dataset.__len__())
    model.eval()
    with torch.no_grad():
        for i, (image, segmentation) in enumerate(loader):
            output = model(image.to(device))
            loss[i] = dice(output, segmentation.to(device))
    model.train()
    return float(torch.mean(loss))


def cross_entropy(prediction, truth):
    """
    Returns the multi-class cross-entropy between prediction and groudn truth. Assumes
    that the input arguments are of the same size.

    """
    eps = 1e-10
    truth = encode_images(truth)
    mask = (1 - truth[:, -1]).unsqueeze(dim=1)
    mask = mask.repeat(1, prediction.shape[1], 1, 1)
    truth = truth[:, 0:-1]
    result = -1 * truth * torch.log(prediction + eps)
    return torch.sum(mask * result) / prediction.shape[0]


def dice(prediction, truth):
    """
    Returns the negative dice coefficient between prediction and ground truth.

    """
    truth = encode_images(truth)
    mask = (1 - truth[:, -1]).unsqueeze(dim=1)
    truth = truth[:, 0:-1]
    numerator = -2 * torch.sum(mask * prediction * truth, dim=(2, 3))
    denominator = torch.sum(prediction**2 + truth**2, dim=(2, 3))
    return torch.mean(numerator / denominator)
