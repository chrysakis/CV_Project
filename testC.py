from utils import decode_image, encode_seg_map
import torch

image = torch.tensor([[[0, 224], [128, 64]],
                      [[0, 224], [128, 128]],
                      [[0, 192], [128, 0]]])
print(image)
seg_map = decode_image(image)
