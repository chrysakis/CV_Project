import torch
from model import UNet

n = 1
d = 572
tensorA = torch.randn((n, 3, d, d))

model = UNet(scale=16)

output = model(tensorA)
print(output.shape)
