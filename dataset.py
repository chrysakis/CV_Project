import os
from PIL import Image
import random
from torch.autograd import Variable
from torchvision.transforms import Compose, Resize, RandomAffine, Pad, \
                                   CenterCrop, RandomHorizontalFlip, ToTensor


class SegmentationDataset:
    def __init__(self, root, year='2009', image_set='all', transform=None):
        self.root = root
        self.year = year
        self.image_set = image_set
        if transform is not None:
            self.transform_output = transform
            transform_b = Compose(transform.transforms[:-2] + [transform.transforms[-1]])
            self.transform_input = transform_b
        else:
            self.transform_input, self.transform_output = None, None
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
        image = Image.open(self.images[index])
        segmentation = Image.open(self.masks[index])
        if self.transform_input is not None:
            seed = random.randint(0, 10**6)
            random.seed(seed)
            image = Variable(self.transform_input(image))
            random.seed(seed)
            segmentation = self.transform_output(segmentation)
            return image, segmentation
        return image, segmentation

    def __len__(self):
        return len(self.images)


transform = Compose([Resize((500, 500)), Pad(100, padding_mode='reflect'),
                     RandomAffine((-10, 10), (0.1, 0.1)), CenterCrop(420),
                     Resize((284, 284)), RandomHorizontalFlip(), Resize((100, 100)),
                     ToTensor()])
