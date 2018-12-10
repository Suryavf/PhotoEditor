#include "Filter.h"

Filter::Filter() {
}

Filter::~Filter() {
	/*for(int i=0; i<5; i++){
		cudaFree(dev_kernel[i]);
	}
	cudaFree(dev_kernel);*/
}

Filter::Filter(double kernel[5*5]) {
    setKernel(kernel);
}

Filter::Filter(double* kernel, int n) {
    setKernel(kernel, 5);
}

bool Filter::setKernel(double kernel[5*5]) {
	setConvolutionKernel2(kernel);
	return true;
}

bool Filter::setKernel(double* kernel, int n) {
	setConvolutionKernel(kernel);
	return true;
}

bool Filter::convolution(uchar* &image, uchar* &result, int x_length, int y_length)
{
    convolutionGPU(image, result, x_length, y_length);
	return true;
}

bool Filter::reserveMemory(double** &matrix, int x, int y) {

	matrix = new double*[x];
	for (int i = 0; i < x; i++) {
		matrix[i] = new double[y];
	}

	ASSERT(matrix != NULL);

	return true;
}

bool Filter::reserveMemory(double* &matrix, int x, int y) {

	matrix = new double[x*y];

	ASSERT(matrix != NULL);

	return true;
}

bool Filter::deleteMemory(double** &matrix, int x, int y) {
	for (int i = 0; i < x; i++) {
		delete[] matrix[i];
	}

	delete[] matrix;

	return true;
}

bool Filter::deleteMemory(double* &matrix, int x, int y) {

	delete[] matrix;

	return true;
}
