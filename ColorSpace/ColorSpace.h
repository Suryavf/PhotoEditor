#ifndef COLOR_SPACE_H
#define COLOR_SPACE_H

#include "../includes.h"

extern "C" void transformColorModel(uchar *h_R , uchar *h_G , uchar *h_B ,
                                    uchar *h_C1, uchar *h_C2, uchar *h_C3,
                                    int n, uint model);

#endif
