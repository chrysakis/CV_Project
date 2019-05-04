import os
from PIL import Image
import random


class SegmentationDataset:
    def __init__(self, root, year='2012', image_set='all', transform=None):
        self.root = root
        self.year = year
        self.image_set = image_set
        self.transform = transform
        voc_root = os.path.join(self.root, 'VOCdevkit/VOC2009/')
        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClass')

        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(split_f):
            raise ValueError('Wrong image_set entered! Please use image_set="train" '
                             'or image_set="trainval" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        input = Image.open(self.images[index])
        output = Image.open(self.masks[index])
        if self.transform is not None:
            seed = random.randint(0, 10**6)
            random.seed(seed)
            input = self.transform(input)
            random.seed(seed)
            output = self.transform(output)
            return input, output
        return input, output

    def __len__(self):
        return len(self.images)
