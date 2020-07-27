
import torch
from torch import nn

from .crop_split_gt_func import crop_split_gt
class CropSplitGt(nn.Module):

    def __init__(self, c=2):
        super(CropSplitGt, self).__init__()
        self.c = c

    def forward(self, data, rois):
        return crop_split_gt(data, rois, self.c)
