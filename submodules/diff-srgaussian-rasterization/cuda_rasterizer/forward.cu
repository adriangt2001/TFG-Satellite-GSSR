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
#include <stdio.h>

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
    int imageSize = sH * sW * CHANNELS;
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
            if (isnan(stdXY)) printf("NaN detected in stdXY at Gaussian %d\n", i);

            float stdX2 = stds[i].x * stds[i].x;
            if (isnan(stdX2)) printf("NaN detected in stdX2 at Gaussian %d\n", i);

            float stdY2 = stds[i].y * stds[i].y;
            if (isnan(stdY2)) printf("NaN detected in stdY2 at Gaussian %d\n", i);

            float beta = 1 - rhos[i] * rhos[i];
            if (isnan(beta)) printf("NaN detected in beta at Gaussian %d\n", i);

            float betaRoot = sqrt(beta);
            if (isnan(betaRoot)) printf("NaN detected in betaRoot at Gaussian %d\n", i);

            float deltaX = (x - means[i].x);
            if (isnan(deltaX)) printf("NaN detected in deltaX at Gaussian %d\n", i);

            float deltaY = (y - means[i].y);
            if (isnan(deltaY)) printf("NaN detected in deltaY at Gaussian %d\n", i);

            float deltaX2 = deltaX * deltaX;
            if (isnan(deltaX2)) printf("NaN detected in deltaX2 at Gaussian %d\n", i);

            float deltaY2 = deltaY * deltaY;
            if (isnan(deltaY2)) printf("NaN detected in deltaY2 at Gaussian %d\n", i);

            float deltaXY = deltaX * deltaY;
            if (isnan(deltaXY)) printf("NaN detected in deltaXY at Gaussian %d\n", i);

            float exp1 = -1 / (2 * beta);
            if (isnan(exp1)) printf("NaN detected in exp1 at Gaussian %d\n", i);

            float exp2 = deltaX2 / stdX2 + deltaY2 / stdY2 - 2 * rhos[i] * deltaXY / stdXY;
            if (isnan(exp2)) printf("NaN detected in exp2 at Gaussian %d\n", i);

            float f = 1 / (2 * M_PI * stdXY * betaRoot) * exp(exp1 * exp2);
            if (isnan(f)) printf("NaN detected in f at Gaussian %d\n", i);

            // Eq. 2
            for (int c = 0; c < CHANNELS; ++c)
            {
                int idx = pixelX * sW * CHANNELS + pixelY * CHANNELS + c;
                
                float op = opacity[i];
                if (isnan(op)) printf("NaN detected in opacity[%d]\n", i);
                if (op == 0.0f) printf("Zero value detected in opacity[%d]\n", i);

                float clr = colors[i * CHANNELS + c];
                if (isnan(colors[i * CHANNELS + c])) printf("NaN detected in colors[%d]\n", i * CHANNELS + c);
                if (clr == 0.0f) printf("Zero value detected in colors[%d]\n", i * CHANNELS + c);

                outImage[idx] += op * f * clr;
                if (isnan(outImage[idx])) printf("NaN detected in updated outImage[%d]\n", idx);
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
