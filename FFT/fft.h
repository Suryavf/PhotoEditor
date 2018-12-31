#ifndef EXECUTE_FFT_H
#define EXECUTE_FFT_H

#include "../includes.h"

extern "C" void executeFFT(uchar *h_R , uchar *h_G , uchar *h_B ,
                           uchar *h_Mag, uint rows, uint cols);

#endif


