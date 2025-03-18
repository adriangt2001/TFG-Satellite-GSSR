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
#include <tuple>

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
    const float rasterRatio)
{
    int numGaussians = means.size(0);
    const int sH = scaleFactor * imageHeight;
    const int sW = scaleFactor * imageWidth;

    torch::Tensor out_color = torch::full({NUM_CHANNELS, sH, sW}, 0.0, torch::kFloat32);

    dim3 grid((sW + BLOCK_X - 1) / BLOCK_X, (sH + BLOCK_Y - 1) / BLOCK_Y);
    dim3 block(BLOCK_X, BLOCK_Y);

    FORWARD::render(
        grid, block,
        numGaussians,
        opacity.contiguous().data_ptr<float>(),
        reinterpret_cast<float2 *>(means.contiguous().data_ptr<float>()),
        reinterpret_cast<float2 *>(stds.contiguous().data_ptr<float>()),
        rhos.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        sH, sW,
        scaleFactor,
        rasterRatio,
        out_color.data_ptr<float>());
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
    const float raster_ratio
)
{
    int numGaussians = means.size(0);
    const int sH = scale_factor * image_height;
    const int sW = scale_factor * image_width;

    torch::Tensor dL_dopacity = torch::zeros_like(opacity);
    torch::Tensor dL_dmeans = torch::zeros_like(means);
    torch::Tensor dL_dstds = torch::zeros_like(stds);
    torch::Tensor dL_drhos = torch::zeros_like(rhos);
    torch::Tensor dL_dcolors = torch::zeros_like(colors);

    dim3 grid((sW + BLOCK_X - 1) / BLOCK_X, (sH + BLOCK_Y - 1) / BLOCK_Y);
    dim3 block(BLOCK_X, BLOCK_Y);

    BACKWARD::render(
        grid, block,
        numGaussians,
        opacity.contiguous().data_ptr<float>(),
        reinterpret_cast<float2 *>(means.contiguous().data_ptr<float>()),
        reinterpret_cast<float2 *>(stds.contiguous().data_ptr<float>()),
        rhos.contiguous().data_ptr<float>(),
        colors.contiguous().data_ptr<float>(),
        grad_output.contiguous().data_ptr<float>(),
        sH, sW,
        scale_factor,
        raster_ratio,
        dL_dopacity.data_ptr<float>(),
        reinterpret_cast<float2 *>(dL_dmeans.data_ptr<float>()),
        reinterpret_cast<float2 *>(dL_dstds.data_ptr<float>()),
        dL_drhos.data_ptr<float>(),
        dL_dcolors.data_ptr<float>()
    );

    return std::make_tuple(dL_dopacity, dL_dmeans, dL_dstds, dL_drhos, dL_dcolors);
}
