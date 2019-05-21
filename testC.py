from utils import encode_images, decode_seg_maps
import torch

image = torch.tensor([[[0, 224], [128, 64]],
                      [[0, 224], [128, 128]],
                      [[0, 192], [128, 0]]])
print(image)
seg_map = encode_images(image)
