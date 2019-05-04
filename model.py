import torch
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, MaxPool2d, BatchNorm2d, ReLU
from torch.nn.functional import pad


class UNet:
    def __init__(self):
        pass

    def forward(self):
        pass


class Contracting(Module):
    def __init__(self, size_in, size_out):
        super(Contracting, self).__init__()

        self.layer_one = Sequential(Conv2d(size_in, size_out, kernel_size=3),
                                    BatchNorm2d(size_out),
                                    ReLU(),
                                    Conv2d(size_out, size_out, kernel_size=3),
                                    BatchNorm2d(size_out),
                                    ReLU())

        self.layer_two = Sequential(Conv2d(size_out, size_out, kernel_size=3),
                                    BatchNorm2d(size_out),
                                    ReLU(),
                                    Conv2d(size_out, size_out, kernel_size=3),
                                    BatchNorm2d(size_out),
                                    ReLU())

    def forward(self, x):
        x = self.layer_one(x)
        return self.layer_two(x)


class Expanding(Module):
    def __init__(self, size_in, size_out):
        super(Expanding, self).__init__()

        self.up = ConvTranspose2d(size_in, size_out, kernel_size=2, stride=2)
        self.conv = Contracting(size_in, size_out)   # BatchNorm may not be needed here

    def forward(self, x1, x2):
        # TODO: Rename and check if padding is correct
        y2 = self.up(x2)
        diff = (y2.shape[2] - x1.shape[2]) // 2
        y1 = pad(x1, 4 * [diff])
        return self.conv(torch.cat((y1, y2), dim=1))
