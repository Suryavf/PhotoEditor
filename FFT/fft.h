#ifndef EXECUTE_FFT_H
#define EXECUTE_FFT_H

#include <stdio.h>
#include <math.h>

extern "C" void executeFFT(uchar *h_R , uchar *h_G , uchar *h_B ,
                           uchar *h_Mag, int rows, int cols);

#endif


