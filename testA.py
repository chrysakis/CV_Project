from utils import encode_images, decode_seg_maps
from dataset import SegmentationDataset
import matplotlib.pyplot as plt
import numpy.random as random
from torchvision.transforms import Compose, Resize, RandomAffine, Pad, \
                                   CenterCrop, RandomHorizontalFlip, ToTensor, ToPILImage

transform = Compose([Resize((500, 500)), Pad(100, padding_mode='reflect'),
                     RandomAffine((-10, 10), (0.1, 0.1)), CenterCrop(420),
                     Resize((280, 280)), RandomHorizontalFlip(), ToTensor()])
dataset = SegmentationDataset(root='../data', year='2009', image_set='train',
                              transform=transform)
pil = ToPILImage()
for index in set(random.randint(0, 1498, 5000)):
    image, segmentation = dataset.__getitem__(index)
    fig = plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.subplot(1, 2, 2)
    seg_map = encode_images(segmentation.unsqueeze(dim=0))
    segmentation = decode_seg_maps(seg_map)
    plt.imshow(segmentation[0])
    plt.show()
