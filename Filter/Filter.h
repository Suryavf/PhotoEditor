#ifndef FILTER_H
#define FILTER_H

#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <omp.h>
#include <assert.h>

#define ASSERT assert
#define dim_kernel 5
#define size_kernel dim_kernel*dim_kernel
typedef unsigned char uchar;

class Filter {

private:
	double** mkernel;
	int klength = dim_kernel;

public:
	Filter();
    Filter(const double kernel[][dim_kernel]);
	Filter(double** kernel, int n);
	~Filter();

	bool setKernel(const double kernel[][dim_kernel]);
	bool setKernel(double** kernel, int n);
    bool convolution(uchar* image, uchar* result, int x_length, int y_length, int thread_count);

	static bool deleteMemory(double** &matrix, int x, int y);
	static bool reserveMemory(double** &matrix, int x, int y);

};

#endif
