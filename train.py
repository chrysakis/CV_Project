import torch
from dataset import SegmentationDataset, transform, transform_test
from torch.utils.data import DataLoader
from model import UNet
from torch.optim import Adam
from loss_functions import cross_entropy, dice, evaluate
import time
import matplotlib.pyplot as plt
from utils import decode_seg_maps, encode_images, plot_losses

# Parameters
batch_size = 16
learning_rate = 5e-2
loss_function = cross_entropy
scale = 8
epochs = 400

# Initialize data, model and optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = SegmentationDataset(root='.', year='2009', image_set='train', transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
unet = UNet(scale=scale).to(device)
optimizer = Adam(unet.parameters(), lr=learning_rate)
train_loss = []
val_loss = []

# Training loop
unet.train()
for epoch in range(epochs):
    start = time.time()
    train_loss.append(0)
    for i, (images, segmentations) in enumerate(loader):
        output = unet(images.to(device))
        loss = loss_function(output, segmentations.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss[-1] += float(loss)
    train_loss[-1] = train_loss[-1] / (i + 1)
    val_loss.append(evaluate(unet, datapath='.', split='val', device=device))
    # clear_output()
    plot_losses(train_loss, val_loss)
    if epoch == 0:
        print(f"Time per epoch: {time.time() - start:.0f} s")

# Plot some examples
dataset = SegmentationDataset(root='../data', year='2009', image_set='train',
                              transform=transform_test)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
unet.eval()
with torch.no_grad():
    for (images, segmentations) in loader:
        predictions = unet(images)
        predictions = decode_seg_maps(predictions)
        segmentations = decode_seg_maps(encode_images(segmentations))
        for i, image in enumerate(images):
            fig = plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(image.permute(1, 2, 0).numpy())
            plt.subplot(1, 3, 2)
            plt.imshow(segmentations[i])
            plt.subplot(1, 3, 3)
            plt.imshow(predictions[i])
            plt.show()
