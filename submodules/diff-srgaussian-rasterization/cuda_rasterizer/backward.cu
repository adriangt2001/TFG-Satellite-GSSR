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

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X *BLOCK_Y)
    renderCUDA(

    )
{
}

void BACKWARD::render(
    const dim3 grid, dim3 block,

)
{
    renderCUDA<NUM_CHANNELS><<<grid, block>>>(

    );
}
