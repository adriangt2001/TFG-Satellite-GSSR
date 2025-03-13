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

#include "rasterizer.h"
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "config.h"
#include "forward.h"
#include "backward.h"

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(bool debug)
{
    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Let each tile blend its range of Gaussians independently in parallel
    const float *feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
    CHECK_CUDA(FORWARD::render(), debug)

    return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(bool debug)
{
    const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Compute loss gradients
    CHECK_CUDA(BACKWARD::render(), debug);
}