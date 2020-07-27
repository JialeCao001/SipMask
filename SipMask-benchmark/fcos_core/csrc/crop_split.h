// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#endif


// Interface for Python
void crop_split_forward(const at::Tensor data,
  const at::Tensor bbox, at::Tensor out,
  const int height,const int width,const int c,const int n)
{
  if (data.type().is_cuda()) {
#ifdef WITH_CUDA
    return crop_split_cuda_forward(data, bbox,  out, height, width, c, n);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}


void crop_split_backward(const at::Tensor data,
  const at::Tensor bbox, at::Tensor out,
  const int height,const int width,const int c, const int n)
{
  if (data.type().is_cuda()) {
#ifdef WITH_CUDA
    return crop_split_cuda_backward(data, bbox, out, height, width, c, n);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}
