#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>

#include<opencv2/core/core.hpp>
#include<opencv2/ml/ml.hpp>
#include<opencv/cv.h>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/video/background_segm.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/imgcodecs.hpp>

typedef unsigned char uchar;
typedef unsigned  int uint ;

void geometricTransformation(uchar *R , uchar *G , uchar *B, uint rows, uint cols);
