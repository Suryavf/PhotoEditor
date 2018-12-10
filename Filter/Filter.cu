#include <cuda_runtime.h>

#include <stdio.h>
#include "Constants.h"
#include "Filter.h"

__constant__ double dev_kernel[KERNEL_LENGTH*KERNEL_LENGTH];

extern "C" void setConvolutionKernel(double* h_Kernel){
    cudaMemcpyToSymbol(dev_kernel, h_Kernel, KERNEL_LENGTH*KERNEL_LENGTH*sizeof(double));
}

extern "C" void setConvolutionKernel2(double h_Kernel[KERNEL_LENGTH*KERNEL_LENGTH]){
    cudaMemcpyToSymbol(dev_kernel, h_Kernel, KERNEL_LENGTH*KERNEL_LENGTH*sizeof(double));
}

__global__ void runConvolutionGPU(uchar* image, uchar* result, int height, int width){
    /*
    int tx = threadIdx.x;
    int ty = threadIdx.y;
	int row_o = threadIdx.y + blockIdx.y*O_TILE_HEIGHT;
	int col_o = threadIdx.x + blockIdx.x*O_TILE_WIDTH;

	int row_i = row_o - KERNEL_LENGTH/2;
	int col_i = col_o - KERNEL_LENGTH/2;

	__shared__ double N_ds[BLOCK_DIM_Y][BLOCK_DIM_X];

	if((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < height)){
        N_ds[ty][tx] = (double)image[row_i*width+col_i];
	}else{
		N_ds[ty][tx] = 0.0f;
	}

	__syncthreads();

	double output = 0.0f;
	if(ty < O_TILE_HEIGHT && tx < O_TILE_WIDTH){
		for(int i=0; i<KERNEL_LENGTH; i++){
			for(int j=0; j<KERNEL_LENGTH; j++){
				output += dev_kernel[i*KERNEL_LENGTH+j]*N_ds[(i+ty)][(j+tx)];
			}
		}
		if(row_o < height && col_o < width){
            result[row_o*width+col_o] = (uchar)output;
		}
	}
    */
    __shared__ double N_ds[BLOCK_DIM_Y][BLOCK_DIM_X];

      // First batch loading
      int dest = threadIdx.y * O_TILE_HEIGHT + threadIdx.x,
         destY = dest / BLOCK_DIM_Y, destX = dest % BLOCK_DIM_X,
         srcY = blockIdx.y * O_TILE_HEIGHT + destY - Mask_radius,
         srcX = blockIdx.x * O_TILE_WIDTH  + destX - Mask_radius,
         src = (srcY * width + srcX) ;
      if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
         N_ds[destY][destX] = I[src];
      else
         N_ds[destY][destX] = 0;

      // Second batch loading
      dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
      destY = dest / BLOCK_DIM_Y, destX = dest % BLOCK_DIM_X;
      srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
      srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
      src = (srcY * width + srcX);
      if (destY < BLOCK_DIM_Y) {
         if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = (double)image[src];
         else
            N_ds[destY][destX] = 0;
      }
      __syncthreads();

      double accum = 0;
      int y, x;
      for (y = 0; y < Mask_width; y++)
         for (x = 0; x < Mask_width; x++)
            accum += N_ds[threadIdx.y + y][threadIdx.x + x] * dev_kernel[y * Mask_width + x];
      y = blockIdx.y * TILE_WIDTH + threadIdx.y;
      x = blockIdx.x * TILE_WIDTH + threadIdx.x;
      if (y < height && x < width)
         result[(y * width + x)  ] = (uchar)clamp(accum);
      __syncthreads();
}

extern "C" void convolutionGPU(uchar* image, uchar* result, int x_length, int y_length){
    dim3 blocks(y_length/O_TILE_HEIGHT + (((y_length%O_TILE_HEIGHT)==0)?0:1), x_length/O_TILE_HEIGHT + (((y_length%O_TILE_HEIGHT)==0)?0:1));
	dim3 threads(BLOCK_DIM_Y,BLOCK_DIM_X);
    runConvolutionGPU<<<blocks,threads>>>(image, result, y_length, x_length);
}
