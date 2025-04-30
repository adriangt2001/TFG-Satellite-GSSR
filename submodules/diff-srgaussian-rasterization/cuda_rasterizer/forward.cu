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
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
    renderCUDA(
        const int numGaussians,
        const float *__restrict__ opacity,
        const float2 *__restrict__ means,
        const float2 *__restrict__ stds,
        const float *__restrict__ rhos,
        const float *__restrict__ colors,
        const int sH,
        const int sW,
        const float scaleFactor,
        const float rasterRatio,
        float *__restrict__ outImage)
{
    // Make this kernel to be per gaussian instead of per pixel

    extern __shared__ float sharedPatch[];

    // Get all Gaussian Parameters and necessary variables for Eq. 1 and 2
    int gaussianIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gaussianIdx >= numGaussians)
        return;

    float alfa = opacity[gaussianIdx];
    float stdX = stds[gaussianIdx].x;
    float stdY = stds[gaussianIdx].y;
    float stdXY = stdX * stdY;
    float stdX2 = stdX * stdX;
    float stdY2 = stdY * stdY;
    float meanX = means[gaussianIdx].x;
    float meanY = means[gaussianIdx].y;
    float rho = rhos[gaussianIdx];
    float beta = 1 - rho * rho;
    float betaRoot = sqrt(beta);
    float exp1 = -1 / (2 * beta);

    float rsH = rasterRatio * sH;
    float rsW = rasterRatio * sW;

    // Iterate through all pixels of the image
    for (int patchX = 0; patchX < sH / 16; patchX++)
    {
        for (int patchY = 0; patchY < sW / 16; patchY++)
        {
            // INIT SHARED MEMORY (16 * 16 * 3!!!!)
            for (int i = 0; i < CHANNELS; i++)
            {
                sharedPatch[threadIdx.x + i * 256] = 0;
                // el último thread tiene trabajo extra
                // para iniciar los valores de la memoria compartida
                // que corresponderia a los threads con id > numGaussians
                // que hacen return arriba
                // esto CASI no causa divergencias ya que ningun thread entra en el if menos el ultimo
                if (gaussianIdx == numGaussians - 1)
                {
                    for (int j = threadIdx.x; j < 256; j++)
                    {
                        sharedPatch[j + i * 256] = 0;
                    }
                }
            }

            __syncthreads();

            for (int pix_x = 0; pix_x < 16; pix_x++)
            {
                for (int pix_y = 0; pix_y < 16; pix_y++)
                {

                    int x = patchX * 16 + pix_x;
                    int y = patchY * 16 + pix_y;

                    // Get pixel coordinates and check if pixel is within the Gaussian influence
                    float deltaX = (x - meanX);
                    float deltaY = (y - meanY);
                    if (fabsf(deltaX) >= rsH || fabsf(deltaY) >= rsW)
                        continue;

                    // Finish computing Eq. 1
                    float deltaX2 = deltaX * deltaX;
                    float deltaY2 = deltaY * deltaY;
                    float deltaXY = deltaX * deltaY;
                    float exp2 = deltaX2 / stdX2 + deltaY2 / stdY2 - 2 * rho * deltaXY / stdXY;
                    float f = 1 / (2 * M_PI * stdXY * betaRoot) * exp(exp1 * exp2);

                    // Eq. 2
                    for (int c = 0; c < CHANNELS; c++)
                    {
                        int idx = pix_x * 16 * CHANNELS + pix_y * CHANNELS + c;
                        float color = colors[gaussianIdx * CHANNELS + c];
                        atomicAdd(&sharedPatch[idx], alfa * color * f);
                    }
                }
            }

            int localPixelX = threadIdx.x % 16;
            int localPixelY = threadIdx.x / 16;
            int globalPixelX = patchX * 16 + localPixelX;
            int globalPixelY = patchY * 16 + localPixelY;

            __syncthreads();

            for (int c = 0; c < CHANNELS; c++)
            {
                int idx = globalPixelX * sW * CHANNELS + globalPixelY * CHANNELS + c;
                int shared_idx = localPixelX * 16 * CHANNELS + localPixelY * CHANNELS + c;
                float color = sharedPatch[shared_idx];
                atomicAdd(&outImage[idx], color);
            }
        }
    }
}

void FORWARD::render(
    const dim3 grid, dim3 block,
    const int numGaussians,
    const float *__restrict__ opacity,
    const float2 *__restrict__ means,
    const float2 *__restrict__ stds,
    const float *__restrict__ rhos,
    const float *__restrict__ colors,
    const int sH,
    const int sW,
    const float scaleFactor,
    const float rasterRatio,
    float *__restrict__ outImage)
{
    renderCUDA<NUM_CHANNELS><<<grid, block, 16 * 16 * NUM_CHANNELS * sizeof(float)>>>(
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
