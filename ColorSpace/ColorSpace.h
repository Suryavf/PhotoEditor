#ifndef COLOR_SPACE_H
#define COLOR_SPACE_H

#include <stdio.h>
#include <math.h>

typedef unsigned char uchar;
typedef unsigned  int uint ;

extern "C" void transformColorModel(uchar *h_R , uchar *h_G , uchar *h_B ,
                                    uchar *h_C1, uchar *h_C2, uchar *h_C3,
                                    int n, uint model);

#endif
