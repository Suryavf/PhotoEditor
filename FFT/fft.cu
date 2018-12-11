#include "fft.h"
#include <omp.h>

// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

typedef cufftDoubleComplex Complex;

void swap(uchar &a, uchar &b){
    uchar aux;
    aux =   a;
    a   =   b;
    b   = aux;
}

uint idx(uint i, uint j, uint cols){
    return  j + i*cols;
}

extern "C" void executeFFT(uchar *h_R , uchar *h_G , uchar *h_B ,
                           uchar *h_Mag, uint rows, uint cols){
    uint i,j, id;
    uint length = rows*cols;

	// RGB to complex
	id = 0;
    Complex **cImg=new Complex*[rows];
#   pragma omp parallel for num_threads(4)
	for (i=0;i<rows;i++){
        cImg[i] = new Complex[rows];
        for (j=0;j<cols;j++){ 
            cImg[i][j].x = (    (double)h_R[id]
                              + (double)h_G[id]
                              + (double)h_B[id]  )/3.0;
            cImg[i][j].y = 0.0;
            ++id;
        }
    }
    
    // Copy to device
    Complex  *d_cImg;
    cudaMalloc((void**) &d_cImg, length*sizeof(Complex));

    for(i=0; i<rows; ++i){
        cudaMemcpy2D(d_cImg + i*cols, sizeof(Complex), cImg[i],
                                      sizeof(Complex), sizeof(Complex), cols,
                                      cudaMemcpyHostToDevice);
    }

	// Create plan
    cufftHandle  planFFT;
    cufftPlan2d(&planFFT, rows, cols, CUFFT_Z2Z);
    
    // Execute FFT
    cufftExecZ2Z(planFFT,d_cImg, d_cImg, CUFFT_FORWARD);

    // Copy to host
    Complex *fft = (Complex*)malloc(rows*cols*sizeof(Complex));
    cudaMemcpy(fft, d_cImg, sizeof(Complex)*rows*cols , cudaMemcpyDeviceToHost);

    // Calculate magnitude
    double *mag = (double*)malloc(rows*cols*sizeof(double));
    double minMag =  999999999999999999999.9999,
           maxMag = -999999999999999999999.9999;
    double x,y, _mag;
    double suma = 0;

    id = 0;
#   pragma omp parallel for num_threads(4)
    for (id=0;id<length;++id){
    	x = fft[id].x;
    	y = fft[id].y;

        _mag = log( x*x + y*y ); //sqrt( x*x + y*y);// log( x*x + y*y );
        if(_mag < minMag) minMag = _mag;
        if(_mag > maxMag) maxMag = _mag;

        mag[id] = _mag;
        suma += _mag;
    }

    std::cout << "max:" << maxMag << "\t min:" << minMag << "\t sum:" << suma/length << std::endl;

    // Magnitude
    double aux = 255.0/(maxMag-minMag);
#   pragma omp parallel for num_threads(4)
    for (id=0;id<length;++id){
        h_Mag[id] = ((uchar)(  (mag[id]-minMag)*aux  ));
    }

    // Shift
    uint cx = rows/2;
    uint cy = cols/2;
#   pragma omp parallel for num_threads(4)
    for (i=0;i<cx;i++){
        for (j=0;j<cy;j++){
            swap( h_Mag[ idx(i   ,j   ,cols) ],
                  h_Mag[ idx(i+cx,j+cy,cols) ] );
        }
        for (j=cy;j<cols;j++){
            swap( h_Mag[ idx(i   ,j   ,cols) ],
                  h_Mag[ idx(i+cx,j-cy,cols) ] );
        }
    }

    // Free memory
        free(   fft);
        free(   mag);
        free(  cImg);
    cudaFree(d_cImg);
    cufftDestroy(planFFT);

}

