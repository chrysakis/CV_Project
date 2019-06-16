import torch
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, MaxPool2d, \
                     BatchNorm2d, ReLU, Softmax
from torch.nn.functional import pad


class UNet(Module):
    def __init__(self, scale=64):
        super(UNet, self).__init__()
        sizes = [64, 128, 256, 512, 1024]
        sizes = [size // scale for size in sizes]

        # Contracting part of the network
        self.contracting_1 = Contracting(3, sizes[0], max_pool=True)
        self.contracting_2 = Contracting(sizes[0], sizes[1], max_pool=True)
        self.contracting_3 = Contracting(sizes[1], sizes[2], max_pool=True)
        self.contracting_4 = Contracting(sizes[2], sizes[3], max_pool=True)
        self.contracting_5 = Contracting(sizes[3], sizes[4], max_pool=False)

        # Expanding part of the network
        self.expanding_4 = Expanding(sizes[4], sizes[3])
        self.expanding_3 = Expanding(sizes[3], sizes[2])
        self.expanding_2 = Expanding(sizes[2], sizes[1])
        self.expanding_1 = Expanding(sizes[1], sizes[0])

        # Final layer
        self.final = Sequential(Conv2d(sizes[0], 2, kernel_size=1), Softmax(dim=1))

    def forward(self, x):
        # Contracting part of the network
        # print(f"x: {x.shape}")
        x1 = self.contracting_1(x)
        # print(f"x1: {x1.shape}")
        x2 = self.contracting_2(x1)
        # print(f"x2: {x2.shape}")
        x3 = self.contracting_3(x2)
        # print(f"x3: {x3.shape}")
        x4 = self.contracting_4(x3)
        # print(f"x4: {x4.shape}")
        x5 = self.contracting_5(x4)
        # print(f"x5: {x5.shape}")

        # Expanding part of the network
        z4 = self.expanding_4(x4, x5)
        # print(f"z4: {z4.shape}")
        z3 = self.expanding_3(x3, z4)
        # print(f"z3: {z3.shape}")
        z2 = self.expanding_2(x2, z3)
        # print(f"z2: {z2.shape}")
        z1 = self.expanding_1(x1, z2)
        # print(f"z1: {z1.shape}")

        # Final layer
        return self.final(z1)


class Contracting(Module):
    def __init__(self, size_in, size_out, max_pool):
        super(Contracting, self).__init__()

        self.layer = Sequential(Conv2d(size_in, size_out, kernel_size=3),
                                BatchNorm2d(size_out),
                                ReLU(),
                                Conv2d(size_out, size_out, kernel_size=3),
                                BatchNorm2d(size_out),
                                ReLU())

        self.max_pool = max_pool
        self.pooling = MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.layer(x)
        if self.max_pool:
            return self.pooling(x)
        else:
            return x


class Expanding(Module):
    def __init__(self, size_in, size_out):
        super(Expanding, self).__init__()

        self.up = ConvTranspose2d(size_in, size_out, kernel_size=2, stride=2)
        self.conv = Contracting(size_in, size_out, max_pool=False)

    def forward(self, x1, x2):
        y2 = self.up(x2)
        diff = (y2.shape[2] - x1.shape[2]) // 2
        y1 = pad(x1, 4 * [diff])
        return self.conv(torch.cat((y1, y2), dim=1))
