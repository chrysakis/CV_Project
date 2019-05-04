import torch
from torch.nn import Sequential, Conv2d, ConvTranspose2d, BatchNorm2d, ReLU
import time

device = 'cpu'
size = 128
batch_size = 10
x = torch.randn((batch_size, 3, size, size)).to(device)
print(x.shape)
contracting = Sequential(Conv2d(3, 64, 3, 1), BatchNorm2d(64), ReLU(),
                         Conv2d(64, 64, 3, 1), BatchNorm2d(64), ReLU()).to(device)
start = time.time()
for i in range(10):
    y = contracting(x)
end = time.time()
print(y.shape)
print(f"\nTime: {(end - start) / 10:.3f} s")
