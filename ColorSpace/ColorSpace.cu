// CUDA runtime
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Utilities and system includes
//#include <helper_functions.h>
//#include <helper_cuda.h>

#include "ColorSpace.h"

/*
 *  LMS
 *  ---
 *  http://biecoll.ub.uni-bielefeld.de/volltexte/2007/52/pdf/ICVS2007-6.pdf
 */
__global__ void rgb2lms(uchar *R, uchar *G, uchar *B, 
                        uchar *L, uchar *M, uchar *S, int *N) {
    int index;
    index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(index < N[0]){
        float r, g, b;
        uchar l, m, s;
 
        r = (float)R[index];
        g = (float)G[index];
        b = (float)B[index];
         
        l = (uchar)(17.8824*r + 43.5161*g + 4.1193*b);
        m = (uchar)( 3.4557*r + 27.1554*g + 3.8671*b);
        s = (uchar)(0.02996*r + 0.18431*g + 1.4670*b);
         
        L[index] = l;
        M[index] = m;
        S[index] = s;
    }
}

/*
 *  XYZ (Adobe RGB [1998])
 *  ----------------------
 *  http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
 */
__global__ void rgb2xyz(uchar *R, uchar *G, uchar *B, 
                        uchar *X, uchar *Y, uchar *Z, int *N) {
    int index;
    index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(index < N[0]){
        float r, g, b;
        uchar x, y, z;

        r = (float)R[index];
        g = (float)G[index];
        b = (float)B[index];
         
        x = (uchar)(0.5767309*r + 0.1855540*g + 0.1881852*b);
        y = (uchar)(0.2973769*r + 0.6273491*g + 0.0752741*b);
        z = (uchar)(0.0270343*r + 0.0706872*g + 0.9911085*b);
         
        X[index] = x;
        Y[index] = y;
        Z[index] = z;
    }
}

/*
 *  CMY
 *  ---
 */
__global__ void rgb2cmy(uchar *R, uchar *G, uchar *B, 
                        uchar *C, uchar *M, uchar *Y, int *N) {
    int index;
    index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(index < N[0]){
        uchar r, g, b;
        uchar c, m, y;
 
        r = R[index];
        g = G[index];
        b = B[index];
         
        c = 255 - r;
        m = 255 - g;
        y = 255 - b;
         
        C[index] = c;
        M[index] = m;
        Y[index] = y;
    }
}

/*
 *  HSL
 *  ---
 */
__global__ void rgb2hsl(uchar *R, uchar *G, uchar *B, 
                        uchar *H, uchar *S, uchar *L, int *N) {
    int index;
    index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(index < N[0]){
        float r, g, b, min, max, c;
        float h, s, l;

        r = (float)R[index];
        g = (float)G[index];
        b = (float)B[index];
        
        // min/max calculate
        max = fmaxf( r ,g);
        max = fmaxf(max,b);

        min = fminf( r ,g);
        min = fminf(min,b);

        c = max - min;

        // Hue and chroma
        if      ( max == r ){ h = fmodf((g-b)/c , 6.0);
        }else if( max == g ){ h =       (b-r)/c + 2.0 ;
        }else if( max == b ){ h =       (r-g)/c + 4.0 ;
        }else               { h = 0.0f;
        }
        h = h*255/6;

        // Lightness
        l = (max + min)/2.0;

        // Saturation
        s = 1.0 - fabsf(2.0*l-1.0);
        if (s == 0.0f){
            s = 0.0;
        }else{
            s = c/s;
        }

        H[index] = (uchar)h;
        S[index] = (uchar)s;
        L[index] = (uchar)l;
    }
}

/*
 *  HSV
 *  ---
 */
__global__ void rgb2hsv(uchar *R, uchar *G, uchar *B, 
                        uchar *H, uchar *S, uchar *V, int *N) {
    int index;
    index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(index < N[0]){
        float r, g, b, min, max, c;
        float h, s, v;
        
        r = (float)R[index];
        g = (float)G[index];
        b = (float)B[index];
        
        // min/max calculate
        max = fmaxf( r ,g);
        max = fmaxf(max,b);

        min = fminf( r ,g);
        min = fminf(min,b);

        c = max - min;

        // Hue and chroma
        if      ( max == r ){ h = fmodf((g-b)/c , 6.0);
        }else if( max == g ){ h =       (b-r)/c + 2.0 ;
        }else if( max == b ){ h =       (r-g)/c + 4.0 ;
        }else               { h = 0.0f;
        }
        h = h*255/6;

        // Lightness
        v = max;

        // Saturation
        if (v == 0.0f){
            s = 0.0;
        }else{
            s = c/v;
        }

        H[index] = (uchar)h;
        S[index] = (uchar)s;
        V[index] = (uchar)v;
    }
}

/*
 *  YIQ
 *  ---
 */
__global__ void rgb2yiq(uchar *R, uchar *G, uchar *B, 
                        uchar *Y, uchar *I, uchar *Q, int *N) {
    int index;
    index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(index < N[0]){
        float r, g, b;
        uchar y, i, q;
 
        r = ((float)R[index])/255.0;
        g = ((float)G[index])/255.0;
        b = ((float)B[index])/255.0;
         
        y = (uchar)((0.299*r + 0.587*g + 0.114*b         )*255.0);
        i = (uchar)((0.596*r - 0.274*g - 0.322*b + 0.5957)*127.5);
        q = (uchar)((0.211*r - 0.523*g + 0.312*b + 0.5226)*127.5);
         
        Y[index] = y;
        I[index] = i;
        Q[index] = q;
    }
}

/*
 *  YUV (BT.709)
 *  ------------
 */
__global__ void rgb2yuv(uchar *R, uchar *G, uchar *B, 
                        uchar *Y, uchar *U, uchar *V, int *N) {
    int index;
    index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(index < N[0]){
        float r, g, b;
        uchar y, u, v;
 
        r = (float)R[index];
        g = (float)G[index];
        b = (float)B[index];
         
        y = (uchar)( 0.2126 *r + 0.7152 *g + 0.0722 *b);
        u = (uchar)(-0.09991*r - 0.33609*g + 0.436  *b);
        v = (uchar)( 0.6150 *r - 0.55861*g - 0.05639*b);

        Y[index] = y;
        U[index] = u;
        V[index] = v;
    }
}

/*
 *  YCbCr
 *  -----
 */
__global__ void rgb2yCbCr(uchar *R, uchar *G , uchar *B, 
                          uchar *Y, uchar *Cb, uchar *Cr, int *N) {
    int index;
    index = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(index < N[0]){
        float r, g, b;
        uchar y, cb, cr;
 
        r = (float)R[index];
        g = (float)G[index];
        b = (float)B[index];
         
        y  = (uchar)( 0.299*r + 0.587*g +  0.114*b);
        cb = (uchar)(-0.169*r - 0.331*g +  0.499*b + 128);
        cr = (uchar)( 0.499*r - 0.418*g - 0.0813*b + 128);
         
        Y [index] =  y;
        Cb[index] = cb;
        Cr[index] = cr;
    }
}


/*
 *  Transform color models
 *  ----------------------
 *  Modelos de color:
 *    - 0: CMY       - 4: LMS
 *    - 1: HSL       - 5: YIQ
 *    - 2: HSV       - 6: YUV
 *    - 3: XYZ       - 7: YCbCr
 */
extern "C" void transformColorModel(uchar *h_R , uchar *h_G , uchar *h_B ,
                                    uchar *h_C1, uchar *h_C2, uchar *h_C3,
                                    int n, uint model){
    uchar *d_R , *d_G , *d_B ,
          *d_C1, *d_C2, *d_C3;
    int   *d_n;

    cudaMalloc(&d_R , n * sizeof(uchar));
    cudaMalloc(&d_G , n * sizeof(uchar));
    cudaMalloc(&d_B , n * sizeof(uchar));
    cudaMalloc(&d_C1, n * sizeof(uchar));
    cudaMalloc(&d_C2, n * sizeof(uchar));
    cudaMalloc(&d_C3, n * sizeof(uchar));
    cudaMalloc(&d_n ,     sizeof( int ));

    cudaMemcpy(d_R, h_R, n * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_G, h_G, n * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, &n,  sizeof( int ), cudaMemcpyHostToDevice);

    int Nthreads = 128;
    int Nblocks  = (int)ceil( ((float)n+((float)Nthreads-1.0)) / ((float)Nthreads) );

    switch(model){
        // CMY
        case 0:
            rgb2cmy<<< Nblocks,Nthreads >>>(d_R ,d_G ,d_B ,
                                            d_C1,d_C2,d_C3, d_n);
            break;

        // HSL
        case 1:
            rgb2hsl<<< Nblocks,Nthreads >>>(d_R ,d_G ,d_B ,
                                            d_C1,d_C2,d_C3, d_n);
            break;

        // HSV
        case 2:
            rgb2hsv<<< Nblocks,Nthreads >>>(d_R ,d_G ,d_B ,
                                            d_C1,d_C2,d_C3, d_n);
            break;

        // XYZ
        case 3:
            rgb2xyz<<< Nblocks,Nthreads >>>(d_R ,d_G ,d_B ,
                                            d_C1,d_C2,d_C3, d_n);
            break;

        // LMS
        case 4:
            rgb2lms<<< Nblocks,Nthreads >>>(d_R ,d_G ,d_B ,
                                            d_C1,d_C2,d_C3, d_n);
            break;

        // YIQ
        case 5:
            rgb2yiq<<< Nblocks,Nthreads >>>(d_R ,d_G ,d_B ,
                                            d_C1,d_C2,d_C3, d_n);
            break;

        // YUV
        case 6:
            rgb2yuv<<< Nblocks,Nthreads >>>(d_R ,d_G ,d_B ,
                                            d_C1,d_C2,d_C3, d_n);
            break;

        // YCbCr
        case 7:
            rgb2yCbCr<<< Nblocks,Nthreads >>>(d_R ,d_G ,d_B ,
                                              d_C1,d_C2,d_C3, d_n);
            break;
    }

    cudaMemcpy(h_C1, d_C1, n * sizeof(uchar), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C2, d_C2, n * sizeof(uchar), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C3, d_C3, n * sizeof(uchar), cudaMemcpyDeviceToHost);

    cudaFree(d_n);
    cudaFree(d_R);
    cudaFree(d_G);
    cudaFree(d_B);
    cudaFree(d_C1);
    cudaFree(d_C2);
    cudaFree(d_C3);
}
