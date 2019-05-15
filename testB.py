import torch
from torch.autograd import Variable
from model import UNet

n, d = 5, 220
tensorA = Variable(torch.randn((n, 3, d, d)))

model = UNet(scale=1)

output = model(tensorA)

print(output.shape)
