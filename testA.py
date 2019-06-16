from utils import encode_images, decode_seg_maps
from dataset import SegmentationDataset, transform, transform_test
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

dataset = SegmentationDataset(root='../data', year='2009', image_set='train',
                              transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
for (images, segmentations) in loader:
    segmentations = decode_seg_maps(encode_images(segmentations))
    for i, image in enumerate(images):
        segmentation = segmentations[i]
        fig = plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(image.permute(1, 2, 0).numpy())
        plt.subplot(1, 2, 2)
        plt.imshow(segmentation)
        plt.show()
