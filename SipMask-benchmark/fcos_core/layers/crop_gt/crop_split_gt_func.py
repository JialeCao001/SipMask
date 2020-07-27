import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
from fcos_core import _C

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
        # ctx.rois = rois
        # print(height*width*n)
        output = data.new_zeros(height, width, n)
        _C.crop_split_gt_forward(data, rois, output, height, width, c, n)
        # print('aa',rois[0])

        # ctx.save_for_backward(data,rois)
        # print(torch.max(output_gt))
        # print('aa',output_gt.shape)
        # print(rois.shape)
        # print(data.requires_grad, rois.requires_grad)
        return output

    # @staticmethod
    # @once_differentiable
    # def backward(ctx, grad_output):
    #     print(grad_output.shape)
    #     data,rois = ctx.saved_tensors
    #     c = ctx.c
    #     height = ctx.height
    #     width = ctx.width
    #     n = ctx.n
    #     # rois = ctx.rois
    #     # print('bb', rois[0])
    #     grad_input = torch.zeros((c*c, height, width, n), dtype=grad_output.dtype, device=grad_output.device)
    #     # grad_input = torch.zeros_like(data)
    #     _C.crop_split_gt_backward(grad_output, rois, grad_input, height, width, c, n)
    #     # print(grad_output.requires_grad,grad_input.requires_grad)
    #
    #     return grad_input, None, None

crop_split_gt = CropSplitGtFunction.apply
