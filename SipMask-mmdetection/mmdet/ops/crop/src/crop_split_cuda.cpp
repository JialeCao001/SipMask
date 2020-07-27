#include <torch/extension.h>
#include <ATen/DeviceGuard.h>

#include <cmath>
#include <vector>

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("crop_split_cuda_forward", &crop_split_cuda_forward,
        "crop split cuda forward(CUDA)");
  m.def("crop_split_cuda_backward",
        &crop_split_cuda_backward,
        "crop split cuda backward(CUDA)");
}
