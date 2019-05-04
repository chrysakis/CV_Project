from dataset import SegmentationDataset
import matplotlib.pyplot as plt
import numpy.random as random
from torchvision.transforms import Compose, Resize, RandomAffine, Pad, \
                                   CenterCrop, RandomHorizontalFlip

transform = Compose([Resize((500, 500)), Pad(50, padding_mode='reflect'),
                     RandomAffine((-15, 15), (0.1, 0.1)), Resize((200, 200)),
                     CenterCrop(128), RandomHorizontalFlip()])
dataset = SegmentationDataset(root='../data', year='2009', image_set='train',
                              transform=transform)
for index in set(random.randint(0, 1498, 5000)):
    image, segmentation = dataset.__getitem__(index)
    fig = plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(segmentation)
    plt.show()
