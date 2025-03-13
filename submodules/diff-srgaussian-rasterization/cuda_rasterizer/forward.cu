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

#include "forward.h"
#include "config.h"

#include <math.h>

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
    renderCUDA(
        const int numGaussians,
        const float* __restrict__ opacity,
        const float2* __restrict__ means,
        const float2* __restrict__ stds,
        const float* __restrict__ rhos,
        const float* __restrict__ colors,
        const int sH, int sW,
        const float scaleFactor,
        const float rasterRatio,
        float* __restrict__ outImage)
{
    int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
    int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
    float x = pixelX / scaleFactor;
    float y = pixelY / scaleFactor;

    if (pixelX >= sW || pixelY >= sH)
        return;

    float rsH = rasterRatio * sH;
    float rsW = rasterRatio * sW;
    for (int i = 0; i < numGaussians; ++i)
    {
        if (fabs(pixelX - mean[i].x) < rsW && fabs(pixelY - mean[i].y) < rsH)
        {
            // Eq. 1
            float sdXY = sd[i].x * sd[i].y;
            float sdX2 = sd[i].x * sd[i].x;
            float sdY2 = sd[i].y * sd[i].y;
            float beta = 1 - rho[i] * rho[i];
            float betaRoot = sqrt(beta);
            float deltaX2 = (x - mean[i].x) * (x - mean[i].x);
            float deltaY2 = (y - mean[i].y) * (y - mean[i].y);
            float deltaXY = (x - mean[i].x) * (y - mean[i].y);
            float exp1 = -1 / (2 * beta);
            float exp2 = deltaX2 / sdX2 + deltaY2 / sdY2 - 2 * rho[i] * deltaXY / sdXY;
            float f = 1 / (2 * M_PI * sdXY * betaRoot) * exp(exp1 * exp2);

            // Eq. 2
            for (int c = 0; c < CHANNELS; ++c)
            {
                int idx = (pixelY * sW + pixelX) * CHANNELS + c;
                outImage[idx] += alfa[i] * f * color[i * CHANNELS + c];
            }
        }
    }
}

void FORWARD::render(
    const dim3 grid, dim3 block,
    const int numGaussians,
    const float* __restrict__ opacity,
    const float2* __restrict__ means,
    const float2* __restrict__ stds,
    const float* __restrict__ rhos,
    const float* __restrict__ colors,
    const int sH, int sW,
    const float scaleFactor,
    const float rasterRatio,
    float* __restrict__ outImage)
{
    renderCUDA<NUM_CHANNELS><<<grid, block>>>(
        numGaussians,
        opacity,
        means,
        stds,
        rhos,
        colors,
        sH, sW,
        scaleFactor,
        rasterRatio,
        outImage);
}
