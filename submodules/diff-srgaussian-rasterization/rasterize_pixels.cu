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
#include "cuda_rasterizer/rasterizer.h"
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

std::tuple<>
RasterizeGaussiansBackwardCUDA()
{
    return std::make_tuple();
}
