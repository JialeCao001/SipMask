from torch import nn

from .crop_split_func import crop_split
import torch
class CropSplit(nn.Module):

    def __init__(self, c=2):
        super(CropSplit, self).__init__()
        self.c = c

    def forward(self, data, rois):
        return crop_split(data, rois, self.c)
