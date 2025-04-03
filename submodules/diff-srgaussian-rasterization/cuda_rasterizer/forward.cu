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
        const int sH, 
        const int sW,
        const float scaleFactor,
        const float rasterRatio,
        float* __restrict__ outImage)
{
    int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
    int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
    float x = pixelX / scaleFactor;
    float y = pixelY / scaleFactor;

    if (pixelX >= sH || pixelY >= sW) return;

    float rsH = rasterRatio * sH;
    float rsW = rasterRatio * sW;
    for (int i = 0; i < numGaussians; ++i)
    {
        if (fabs(x - means[i].x) < rsH && fabs(y - means[i].y) < rsW)
        {
            // Eq. 1
            float stdXY = stds[i].x * stds[i].y;
            float stdX2 = stds[i].x * stds[i].x;
            float stdY2 = stds[i].y * stds[i].y;
            float beta = 1 - rhos[i] * rhos[i];
            float betaRoot = sqrt(beta);
            float deltaX = (x - means[i].x);
            float deltaY = (y - means[i].y);
            float deltaX2 = deltaX * deltaX;
            float deltaY2 = deltaY * deltaY;
            float deltaXY = deltaX * deltaY;
            float exp1 = -1 / (2 * beta);
            float exp2 = deltaX2 / stdX2 + deltaY2 / stdY2 - 2 * rhos[i] * deltaXY / stdXY;
            float f = 1 / (2 * M_PI * stdXY * betaRoot) * exp(exp1 * exp2);

            // Eq. 2
            for (int c = 0; c < CHANNELS; ++c)
            {
                int idx = pixelX * sW * CHANNELS + pixelY * CHANNELS + c;
                float alfa = opacity[i];
                float color = colors[i * CHANNELS + c];
                outImage[idx] += alfa * color * f;
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
    const int sH, 
    const int sW,
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
