#ifndef COLOR_SPACE_H
#define COLOR_SPACE_H

// CUDA runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <math.h>

typedef unsigned char uchar;
typedef unsigned  int uint ;

extern "C" void transformColorModel(uchar *h_R , uchar *h_G , uchar *h_B , 
                                    uchar *h_C1, uchar *h_C2, uchar *h_C2, 
                                    uint n, uint model);

#endif