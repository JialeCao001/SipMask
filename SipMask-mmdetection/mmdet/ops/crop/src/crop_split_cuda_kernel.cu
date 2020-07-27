#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>
#include <stdio.h>
#include <math.h>
#include <algorithm>

using namespace at;
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)
const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N)
{
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename scalar_t>
__global__ void CropSplitKernelForward(
    const int count,
    const scalar_t *bottom_data,
    const scalar_t *bottom_rois,
    const int height,
    const int width,
    const int num_cell,
    const int num_box,
    scalar_t *top_data)
{
  CUDA_KERNEL_LOOP(index, count)
  {
    // The output is in order (n, ctop, ph, pw)
    //int pw = index % width;
    //int ph = (index / width) % height;
    //int n = index / width / height;
    int n = index % num_box;
    int pw = (index / num_box) % width;
    int ph = index / num_box / width;
    // [start, end) interval for spatial sampling
    const scalar_t *offset_bottom_rois = bottom_rois + n * 4;
    scalar_t roi_x1 = offset_bottom_rois[0];
    scalar_t roi_y1 = offset_bottom_rois[1];
    scalar_t roi_x2 = offset_bottom_rois[2];
    scalar_t roi_y2 = offset_bottom_rois[3];

    if((pw >= roi_x1) & (ph >= roi_y1) & (pw < roi_x2) & (ph < roi_y2)){

        scalar_t roi_w = (roi_x2-roi_x1+0.1)/num_cell;
        scalar_t roi_h = (roi_y2-roi_y1+0.1)/num_cell;

        int idx_w = (pw-roi_x1)/roi_w;
        int idx_h = (ph-roi_y1)/roi_h;
        int idx = (idx_h*num_cell+idx_w)*count;
        if(index+idx >= count*num_cell*num_cell){
            printf("(%d %d %d %d %d %d %f %f %f %f), ",index+idx, count*num_cell*num_cell, idx_h, idx_w, num_cell, index, roi_w, roi_h, pw-roi_x1, ph-roi_y1);
        }
        top_data[index] = bottom_data[index+idx];
    }
   }
}

void CropSplitForward(const at::Tensor data,
                        const at::Tensor bbox,
                        at::Tensor out,
                        const int height,
                        const int width,
                        const int num_cell,
                        const int num_bbox)
{
  const int count = num_bbox * height * width;
  //printf("aa, %d ",count);
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      data.type(), "CropSplitForward", ([&] {
        const scalar_t *bottom_data = data.data<scalar_t>();
        const scalar_t *bottom_rois = bbox.data<scalar_t>();
        scalar_t *top_data = out.data<scalar_t>();

        CropSplitKernelForward<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
            count, bottom_data, bottom_rois, height, width, num_cell, num_bbox, top_data);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in CropSplitForward: %s\n", cudaGetErrorString(err));
  }
}


template <typename scalar_t>
__global__ void CropSplitKernelBack(
    const int count,
    const scalar_t *top_diff,
    const scalar_t *bottom_rois,
    const int height,
    const int width,
    const int num_cell,
    const int num_box,
    scalar_t *bottom_diff)
{
  CUDA_KERNEL_LOOP(index, count)
  {
    // The output is in order (n, ctop, ph, pw)
    //int pw = index % width;
    //int ph = (index / width) % height;
    //int n = index / width / height;
    int n = index % num_box;
    int pw = (index / num_box) % width;
    int ph = index / num_box / width;
    // [start, end) interval for spatial sampling
    const scalar_t *offset_bottom_rois = bottom_rois + n * 4;
    scalar_t roi_x1 = offset_bottom_rois[0];
    scalar_t roi_y1 = offset_bottom_rois[1];
    scalar_t roi_x2 = offset_bottom_rois[2];
    scalar_t roi_y2 = offset_bottom_rois[3];
    if((pw >= roi_x1) & (ph >= roi_y1) & (pw < roi_x2) & (ph < roi_y2)){
        scalar_t roi_w = (roi_x2-roi_x1+0.1)/num_cell;
        scalar_t roi_h = (roi_y2-roi_y1+0.1)/num_cell;

        int idx_w = (pw-roi_x1)/roi_w;
        int idx_h = (ph-roi_y1)/roi_h;


        int idx = (idx_h*num_cell+idx_w)*count;
        atomicAdd(bottom_diff+index+idx, top_diff[index]);
    }
   }
}


void CropSplitBack(const at::Tensor top_grad,
                    const at::Tensor bbox,
                    at::Tensor bottom_grad,
                    const int height,
                    const int width,
                    const int num_cell,
                    const int num_bbox)
{
  const int count = num_bbox * height * width;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "CropSplitBack", ([&] {
        const scalar_t *top_diff = top_grad.data<scalar_t>();
        const scalar_t *bottom_rois = bbox.data<scalar_t>();
        scalar_t *bottom_diff = bottom_grad.data<scalar_t>();

        CropSplitKernelBack<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
            count, top_diff, bottom_rois, height, width, num_cell, num_bbox, bottom_diff);
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in CropSplitBack: %s\n", cudaGetErrorString(err));
  }
}
