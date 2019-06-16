import torch
import matplotlib.pyplot as plt


def decode_seg_maps(seg_maps):
    """
    Transform a batch of segmentation maps into images ready to be visualized.

    """
    device = seg_maps.device
    n_maps = seg_maps.shape[1]
    seg_maps = seg_maps.permute(1, 0, 2, 3).type(torch.float)
    max_per_channel = torch.max(seg_maps, dim=0, keepdim=True)[0].repeat(n_maps, 1, 1, 1)
    seg_maps = (seg_maps == max_per_channel).type(torch.float).to(device)
    for i, channel in enumerate(seg_maps):
        seg_maps[i] = channel * i
    seg_maps = torch.sum(seg_maps, dim=0)
    seg_maps = seg_maps.type(torch.long)
    colors = get_colors(device)
    return colors[seg_maps].cpu().numpy()


def encode_images(images):
    """
    Transform a batch of images as read by the dataloader into a batch of
    segmentation maps.

    """
    images = 255 * images.permute(1, 0, 2, 3)
    images += (images == 255).type(torch.float) * (21 - 255)
    images = images.repeat(3, 1, 1, 1)
    images[0] = (images[0] == 0).type(torch.float)
    images[1] = (1 <= images[1]).type(torch.float) * (images[1] <= 20).type(torch.float)
    images[2] = (images[2] == 21).type(torch.float)
    return images.permute(1, 0, 2, 3)


def plot_losses(train_loss, val_loss=None):
    epochs = range(1, len(train_loss) + 1)
    plt.figure()
    plt.plot(epochs, train_loss)
    if val_loss is not None:
        plt.plot(epochs, val_loss)
        plt.legend(['Training Loss', 'Validation Loss'])
    else:
        plt.legend(['Training Loss'])
    plt.grid()
    plt.show()


def get_colors(device='cpu'):
    colors = torch.tensor(
        [
            [0, 0, 0],
            [255, 255, 255],
            [128, 128, 128]
        ],
        dtype=torch.uint8)
    return colors.to(device)
