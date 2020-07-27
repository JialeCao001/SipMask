// modify from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/modulated_dcn_cuda.c

// based on
// author: Charles Shang
// https://github.com/torch/cunn/blob/master/lib/THCUNN/generic/SpatialConvolutionMM.cu

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>
#include <cmath>

void CropSplitForward(const at::Tensor data, const at::Tensor bbox, at::Tensor out,
                        const int height, const int width, const int num_cell, const int num_bbox);

void CropSplitBack(const at::Tensor top_grad, const at::Tensor bbox, at::Tensor bottom_grad,
                    const int height, const int width,  const int num_cell, const int num_bbox);


void crop_split_cuda_forward(const at::Tensor input, const at::Tensor bbox, at::Tensor out,
                        const int height, const int width, const int num_cell, const int num_bbox)
{
  AT_CHECK(input.is_contiguous(), "input tensor has to be contiguous");

  CropSplitForward(input, bbox, out, height, width, num_cell, num_bbox);
}

void crop_split_cuda_backward(const at::Tensor out_grad, const at::Tensor bbox, at::Tensor bottom_grad,
                    const int height, const int width,  const int num_cell, const int num_bbox)
{
  AT_CHECK(out_grad.is_contiguous(), "out_grad tensor has to be contiguous");

  CropSplitBack(out_grad, bbox, bottom_grad, height, width, num_cell, num_bbox);
}
