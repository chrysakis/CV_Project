from dataset import SegmentationDataset, transform
from torch.utils.data import DataLoader
from model import UNet
from torch.optim import Adam
from loss_functions import cross_entropy, dice


# Parameters
batch_size = 4
learning_rate = 1e-4
loss_function = cross_entropy
scale = 32

# Initialize data, model and optimizer
dataset = SegmentationDataset(root='../data', year='2009', image_set='train',
                              transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
unet = UNet(scale=scale)
optimizer = Adam(unet.parameters(), lr=learning_rate)

# Training loop
for i, (images, segmentations) in enumerate(loader):
    output = unet(images)
    loss = loss_function(output, segmentations)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
