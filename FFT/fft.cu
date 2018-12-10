// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <cufft.h>
#include <math.h>

typedef cufftDoubleComplex Complex;

extern "C" void executeFFT(uchar *h_R , uchar *h_G , uchar *h_B ,
                           uchar *h_Mag, int rows, int cols){
	int i,j;

	// RGB to complex
	int id = 0;
    Complex **cImg=new Complex*[rows];
	for (i=0;i<rows;i++){
        cImg[i] = new Complex[rows];
        for (j=0;j<cols;j++){
            cImg[i][j].x = (   (double)R[id]
            	              + (double)G[id]
            	              + (double)B[id]  )/3.0;
            cImg[i][j].y = 0.0;
            ++id;
        }
    }
    
    // Copy to device
    Complex  *d_cImg;
    cudaMalloc((void**) &d_cImg, rows*cols*sizeof(Complex));

    for(i=0; i<rows; ++i){
        cudaMemcpy2D(d_cImg + i*cols, sizeof(Complex), cImg[i], sizeof(Complex), sizeof(Complex), cols, cudaMemcpyHostToDevice);
    }

	// Create plan
    cufftHandle  planFFT;
    cufftPlan2d(&planFFT, rows, cols, CUFFT_Z2Z);
    
    // Execute FFT
    cufftExecZ2Z(planFFT,d_cImg, d_cImg, CUFFT_FORWARD);

    // Copy to host
    Complex *fft = (Complex*)malloc(rows*cols*sizeof(Complex));
    cudaMemcpy(fft, d_cImg, sizeof(Complex)*rows*cols , cudaMemcpyDeviceToHost);

    

    

    // Free memory
        free(   fft);
        free(  cImg);
    cudaFree(d_cImg);
    cufftDestroy(planFFT);

}

