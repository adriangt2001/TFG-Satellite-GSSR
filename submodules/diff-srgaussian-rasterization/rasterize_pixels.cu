/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "cuda_rasterizer/config.h"
#include "rasterize_pixels.h"
#include "cuda_rasterizer/forward.h"
#include "cuda_rasterizer/backward.h"
#include <tuple>
#include <iostream>

#define CHECK_CUDA(A, debug) \
A; if(debug) { \
auto ret = cudaDeviceSynchronize(); \
if (ret != cudaSuccess) { \
std::cerr << "\n[CUDA ERROR] in " << __FILE__ << "\nLine " << __LINE__ << ": " << cudaGetErrorString(ret) << std::endl; \
throw std::runtime_error(cudaGetErrorString(ret)); \
} \
}

std::tuple<torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor &opacity,
    const torch::Tensor &means,
    const torch::Tensor &stds,
    const torch::Tensor &rhos,
    const torch::Tensor &colors,
    const int imageHeight,
    const int imageWidth,
    const float scaleFactor,
    const float rasterRatio,
    const bool debug)
{
    int batchSize = means.size(0);
    int numGaussians = means.size(1);

    const int sH = scaleFactor * imageHeight;
    const int sW = scaleFactor * imageWidth;
    // Create output tensor with same device and dtype as opacity
    torch::Tensor out_color = torch::zeros({batchSize, sH, sW, NUM_CHANNELS}, opacity.options());
    
    // Check device of out_color
    std::cout << "out_color device: " << out_color.device() << std::endl;
    std::cout << "means device: " << means.device() << std::endl;

    dim3 grid((sH + BLOCK_X - 1) / BLOCK_X, (sW + BLOCK_Y - 1) / BLOCK_Y);
    dim3 block(BLOCK_X, BLOCK_Y);

    for (int b = 0; b < batchSize; b++) {
        CHECK_CUDA(FORWARD::render(
            grid, block,
            numGaussians,
            opacity.contiguous().data_ptr<float>() + b * numGaussians,
            reinterpret_cast<float2 *>(means.contiguous().data_ptr<float>() + b * numGaussians * 2),
            reinterpret_cast<float2 *>(stds.contiguous().data_ptr<float>() + b * numGaussians * 2),
            rhos.contiguous().data_ptr<float>() + b * numGaussians,
            colors.contiguous().data_ptr<float>() + b * numGaussians * NUM_CHANNELS,
            sH, sW,
            scaleFactor,
            rasterRatio,
            out_color.contiguous().data_ptr<float>() + b * sH * sW * NUM_CHANNELS), debug);
    }
    return std::make_tuple(out_color);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& opacity,
    const torch::Tensor& means,
    const torch::Tensor& stds,
    const torch::Tensor& rhos,
    const torch::Tensor& colors,
    const torch::Tensor& grad_output,
    const int image_height,
    const int image_width,
    const float scale_factor,
    const float raster_ratio,
    const bool debug
)
{
    int batchSize = means.size(0);
    int numGaussians = means.size(1);

    const int sH = scale_factor * image_height;
    const int sW = scale_factor * image_width;

    torch::Tensor dL_dopacity = torch::zeros_like(opacity);
    torch::Tensor dL_dmeans = torch::zeros_like(means);
    torch::Tensor dL_dstds = torch::zeros_like(stds);
    torch::Tensor dL_drhos = torch::zeros_like(rhos);
    torch::Tensor dL_dcolors = torch::zeros_like(colors);

    dim3 grid((sW + BLOCK_X - 1) / BLOCK_X, (sH + BLOCK_Y - 1) / BLOCK_Y);
    dim3 block(BLOCK_X, BLOCK_Y);
    for (int b = 0; b < batchSize; b++) {
        CHECK_CUDA(BACKWARD::render(
            grid, block,
            numGaussians,
            opacity.contiguous().data_ptr<float>() + b * numGaussians,
            reinterpret_cast<float2 *>(means.contiguous().data_ptr<float>() + b * numGaussians * 2),
            reinterpret_cast<float2 *>(stds.contiguous().data_ptr<float>() + b * numGaussians * 2),
            rhos.contiguous().data_ptr<float>() + b * numGaussians,
            colors.contiguous().data_ptr<float>() + b * numGaussians * NUM_CHANNELS,
            grad_output.contiguous().data_ptr<float>() + b * sH * sW * NUM_CHANNELS,
            sH, sW,
            scale_factor,
            raster_ratio,
            dL_dopacity.data_ptr<float>() + b * numGaussians,
            reinterpret_cast<float2 *>(dL_dmeans.data_ptr<float>() + b * numGaussians * 2),
            reinterpret_cast<float2 *>(dL_dstds.data_ptr<float>() + b * numGaussians * 2),
            dL_drhos.data_ptr<float>() + b * numGaussians,
            dL_dcolors.data_ptr<float>() + b * numGaussians * NUM_CHANNELS
        ), debug);
    }
    return std::make_tuple(dL_dopacity, dL_dmeans, dL_dstds, dL_drhos, dL_dcolors);
}
