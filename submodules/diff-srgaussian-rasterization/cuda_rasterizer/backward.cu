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

#include "backward.h"
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
        const float* __restrict__ grad_output,
        const int sH, int sW,
        const float scaleFactor,
        const float rasterRatio,
        float* __restrict__ dL_dopacity,
        float2* __restrict__ dL_dmeans,
        float2* __restrict__ dL_dstds,
        float* __restrict__ dL_drhos,
        float* __restrict__ dL_dcolors
    )
{
    int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
    int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
    float x = pixelX / scaleFactor;
    float y = pixelY / scaleFactor;

    if (pixelX >= sW || pixelY >= sH) return;

    float rsH = rasterRatio * sH;
    float rsW = rasterRatio * sW;
    for (int i = 0; i < numGaussians;  ++i) {
        if (fabs(x - means[i].x) < rsW && fabs(y - means[i].y) < rsH) {
            // Compute Gaussian function value similar to forward pass
            float stdX = stds[i].x;
            float stdY = stds[i].y;
            float stdXY = stdX * stdY;
            float stdX2 = stdX * stdX;
            float stdY2 = stdY * stdY;
            float rho = rhos[i];
            float beta = 1 - rho * rho;
            float betaRoot = sqrt(beta);
            float meanX = means[i].x;
            float meanY = means[i].y;
            float deltaX = (x - meanX);
            float deltaY = (y - meanY);
            float deltaX2 = deltaX * deltaX;
            float deltaY2 = deltaY * deltaY;
            float deltaXY = deltaX * deltaY;
            float exp1 = -1 / (2 * beta);
            float exp2 = deltaX2 / stdX2 + deltaY2 / stdY2 - 2 * rho * deltaXY / stdXY;
            float f = 1 / (2 * M_PI * stdXY * betaRoot) * exp(exp1 * exp2);

            // Now compute gradients
            for (int c = 0; c < CHANNELS; ++c) {
                int idx = (pixelY * sW + pixelX) * CHANNELS + c;
                float grad = grad_output[idx];

                if (grad != 0) { // because if 0, it doesn't affect
                    float color = colors[i * CHANNELS + c];
                    float alfa = opacity[i];

                    // Opacity grad
                    atomicAdd(&dL_dopacity[i], grad * f * color);
                    
                    // Mean X grad
                    float dL_df = grad * alfa * color;
                    float df_dmeanx = f * exp1 * ((-2 * deltaX / stdX2) + (2 * rho * deltaY / stdXY));
                    atomicAdd(&dL_dmeans[i].x, dL_df * df_dmeanx);

                    // Mean Y grad
                    float df_dmeany = f * exp1 * ((-2 * deltaY / stdY2) + (2 * rho * deltaX / stdXY));
                    atomicAdd(&dL_dmeans[i].y, dL_df * df_dmeany);

                    // Std X grad
                    float df_dstdx = -f / stdX + f * exp1 * (-2 * deltaX2 / stdX2 / stdX + 2 * rho * deltaXY / stdXY / stdX);
                    atomicAdd(&dL_dstds[i].x, dL_df * df_dstdx);
                    
                    // Std Y grad
                    float df_dstdy = -f / stdY + f * exp1 * (-2 * deltaY2 / stdY2 / stdY + 2 * rho * deltaXY / stdXY / stdY);
                    atomicAdd(&dL_dstds[i].y, dL_df * df_dstdy);

                    // Rho grad
                    float df_drho = f * ((rho / beta) - (rho * exp2 / (beta * beta)) - (exp1 * 2 * deltaXY / stdXY));
                    atomicAdd(&dL_drhos[i], dL_df * df_drho);

                    // Color grad
                    atomicAdd(&dL_dcolors[i * CHANNELS + c], grad * opacity[i] * f);
                }
            }
        }
    }
}

void BACKWARD::render(
    const dim3 grid, dim3 block,
    const int numGaussians,
    const float* __restrict__ opacity,
    const float2* __restrict__ means,
    const float2* __restrict__ stds,
    const float* __restrict__ rhos,
    const float* __restrict__ colors,
    const float* __restrict__ grad_output,
    const int sH, int sW,
    const float scaleFactor,
    const float rasterRatio,
    float* __restrict__ dL_dopacity,
    float2* __restrict__ dL_dmeans,
    float2* __restrict__ dL_dstds,
    float* __restrict__ dL_drhos,
    float* __restrict__ dL_dcolors
)
{
    renderCUDA<NUM_CHANNELS><<<grid, block>>>(
        numGaussians,
        opacity,
        means,
        stds,
        rhos,
        colors,
        grad_output,
        sH, sW,
        scaleFactor,
        rasterRatio,
        dL_dopacity,
        dL_dmeans,
        dL_dstds,
        dL_drhos,
        dL_dcolors
    );
}
