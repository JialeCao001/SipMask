import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from . import crop_split_gt_cuda

class CropSplitGtFunction(Function):

    @staticmethod
    def forward(ctx, data, rois, c):
        height = data.shape[0]
        width = data.shape[1]
        n = data.shape[2]
        ctx.c = _pair(c)
        ctx.height = _pair(height)
        ctx.width = _pair(width)
        ctx.n = _pair(n)

        output = data.new_zeros(height, width, n)
        crop_split_gt_cuda.crop_split_gt_cuda_forward(data, rois, output, height, width, c, n)

        return output


crop_split_gt = CropSplitGtFunction.apply

class CropSplitGt(nn.Module):

    def __init__(self, c=2):
        super(CropSplitGt, self).__init__()
        self.c = c

    def forward(self, data, rois):
        return crop_split_gt(data, rois, self.c)
