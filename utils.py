import torch


def decode_seg_maps(seg_maps):
    """
    Transform a batch of segmentation maps into images ready to be visualized.

    """
    device = seg_maps.device
    seg_maps = seg_maps.permute(1, 0, 2, 3).type(torch.float)
    for i, channel in enumerate(seg_maps):
        seg_maps[i] = channel * i
    seg_maps = torch.sum(seg_maps, dim=0)
    seg_maps = seg_maps.type(torch.long)
    colors = get_colors(device)
    return colors[seg_maps].to('cpu').numpy()


def encode_images(images):
    """
    Transform a batch of images as read by the dataloader into a batch of
    segmentation maps.

    """
    images = 255 * images.permute(1, 0, 2, 3)
    images += (images == 255).type(torch.float) * (21 - 255)
    images = images.repeat(22, 1, 1, 1)
    for i, channel in enumerate(images):
        images[i] = (channel == i)
    return images.permute(1, 0, 2, 3)


def get_colors(device='cpu'):
    colors = torch.tensor(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
            [224, 224, 192],
        ],
        dtype=torch.uint8)
    return colors.to(device)
