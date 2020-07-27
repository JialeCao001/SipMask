import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from . import crop_split_cuda

class CropSplitFunction(Function):

    @staticmethod
    def forward(ctx, data, rois, c):
        height = data.shape[1]
        width = data.shape[2]
        n = data.shape[3]
        ctx.c = c
        ctx.height = height
        ctx.width = width
        ctx.n = n
        ctx.rois = _pair(rois)
        # print(height*width*n)
        output = data.new_zeros(height, width, n)
        crop_split_cuda.crop_split_cuda_forward(data, rois, output, height, width, c, n)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        c = ctx.c
        height = ctx.height
        width = ctx.width
        n = ctx.n
        rois = ctx.rois
        grad_input = torch.zeros((c*c, height, width, n), dtype=grad_output.dtype, device=grad_output.device)
        crop_split_cuda.crop_split_cuda_backward(grad_output, rois, grad_input, height, width, c, n)

        return grad_input, None, None

crop_split = CropSplitFunction.apply

class CropSplit(nn.Module):

    def __init__(self, c=2):
        super(CropSplit, self).__init__()
        self.c = c

    def forward(self, data, rois):
        return crop_split(data, rois, self.c)