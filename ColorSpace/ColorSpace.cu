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

        l = uchar(0.271873f*r + 0.661593f*g + 0.062627f*b);
        m = uchar(0.099837f*r + 0.784534f*g + 0.111723f*b);
        s = uchar(0.017750f*r + 0.109197f*g + 0.869146f*b);
         
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

        r = float(R[index]);
        g = float(G[index]);
        b = float(B[index]);
         
        x = uchar(0.604410f*r + 0.194460f*g + 0.197220f*b);
        y = uchar(0.296215f*r + 0.624898f*g + 0.074980f*b);
        z = uchar(0.024732f*r + 0.064667f*g + 0.906695f*b);
         
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

        r = float(R[index]);
        g = float(G[index]);
        b = float(B[index]);

        // min/max calculate
        max = fmaxf( r ,g);
        max = fmaxf(max,b);

        min = fminf( r ,g);
        min = fminf(min,b);

        c = max - min;

        // Hue and chroma
              if( max == r ){ h = fmodf((g-b)/c , 6.0f);
        }else if( max == g ){ h =       (b-r)/c + 2.0f ;
        }else if( max == b ){ h =       (r-g)/c + 4.0f ;
        }else               { h =                 0.0f ;
        }
        h = h*255/360;

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
        
        r = float(R[index]);
        g = float(G[index]);
        b = float(B[index]);
        
        // min/max calculate
        max = fmaxf( r ,g);
        max = fmaxf(max,b);

        min = fminf( r ,g);
        min = fminf(min,b);

        c = max - min;

        // Hue and chroma
              if( max == r ){ h = fmodf((g-b)/c , 6.0f);
        }else if( max == g ){ h =       (b-r)/c + 2.0f ;
        }else if( max == b ){ h =       (r-g)/c + 4.0f ;
        }else               { h =                 0.0f ;
        }
        h = h*255/360;

        // Lightness
        v = max;

        // Saturation
        if (v == 0.0f){
            s = 0.0;
        }else{
            s = c*2.55f/v;
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

        r = float(R[index]);
        g = float(G[index]);
        b = float(B[index]);

        y = uchar( 0.29783f*r + 0.58471f*g + 0.11355f*b          );
        i = uchar( 0.49805f*r - 0.22897f*g - 0.26908f*b + 127.5f );
        q = uchar( 0.20093f*r - 0.49805f*g + 0.29711f*b + 127.5f );
         
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

        r = float(R[index]);
        g = float(G[index]);
        b = float(B[index]);

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

        y  = uchar( 0.211770f*r + 0.712406f*g + 0.071918f*b);
        cb = uchar(-0.114130f*r - 0.383920f*g + 0.498050f*b + 127.50f);
        cr = uchar( 0.498047f*r - 0.452380f*g - 0.045666f*b + 127.50f);
         
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
