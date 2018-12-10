#include "Filter.h"

Filter::Filter() {
	klength = dim_kernel;
	reserveMemory(mkernel, klength, klength);
}

Filter::~Filter() {
	deleteMemory(mkernel, klength, klength);
}

Filter::Filter(const double kernel[][dim_kernel]) {
	klength = dim_kernel;// length(kernel);
	if (reserveMemory(mkernel, klength, klength)) {
		setKernel(kernel);
		//std::cout << "Kernel created" << std::endl;
	}
	else {
		//std::cout << "Error in Kernel creation!" << std::endl;
	}
}

Filter::Filter(double** kernel, int n) {
	ASSERT(n == dim_kernel);

	klength = n;
	if (reserveMemory(mkernel, klength, klength)) {
		setKernel(kernel, n);
		//std::cout << "Kernel created" << std::endl;
	}
	else {
		//std::cout << "Error in Kernel creation!" << std::endl;
	}
}

bool Filter::setKernel(const double kernel[][dim_kernel]) {
	klength = dim_kernel;
	for (int i = 0; i < klength; i++) {
		for (int j = 0; j < klength; j++) {
			mkernel[i][j] = kernel[i][j];
		}
	}
	return true;
}

bool Filter::setKernel(double** kernel, int n) {
	ASSERT(n == dim_kernel);

	klength = n;
	for (int i = 0; i < klength; i++) {
		for (int j = 0; j < klength; j++) {
			mkernel[i][j] = kernel[i][j];
		}
	}
	return true;
}

bool Filter::convolution(uchar* image, uchar* result, int x_length, int y_length, int thread_count){
	double** mImage;
	int x_mi_length = x_length + 2*(klength/2);
	int y_mi_length = y_length + 2*(klength / 2);
	reserveMemory(mImage, x_mi_length, y_mi_length);

#pragma omp parallel for num_threads(thread_count) shared(mImage)
	for (int i = 0; i < x_mi_length; i++) {
		std::fill_n(mImage[i], y_mi_length, 0);
	}

	int li_mImage = klength / 2;
	int x_ls_mImage = x_mi_length - li_mImage;
	int y_ls_mImage = y_mi_length - li_mImage;

#pragma omp parallel for collapse(2) num_threads(thread_count) shared(mImage, image)
	for (int i = li_mImage; i < x_ls_mImage; i++) {
		for (int j = li_mImage; j < y_ls_mImage; j++) {
            mImage[i][j] = (double)image[ (i - 2)*y_length + j - 2];
		}
	}

	//Cuadrado central
#pragma omp parallel for collapse(2) num_threads(thread_count) shared(mkernel, mImage, result, thread_count)
	for (int i = li_mImage; i < x_ls_mImage; ++i) {
		for (int j = li_mImage; j < y_ls_mImage; ++j) {
			double acumulador = 0;
			double* krow;
			double* irow;

			krow = mkernel[0];
			irow = mImage[i + 2];
			acumulador += krow[0] * irow[j + 2];
			acumulador += krow[1] * irow[j + 1];
			acumulador += krow[2] * irow[j + 0];
			acumulador += krow[3] * irow[j + -1];
			acumulador += krow[4] * irow[j + -2];

			krow = mkernel[1];
			irow = mImage[i + 1];
			acumulador += krow[0] * irow[j + 2];
			acumulador += krow[1] * irow[j + 1];
			acumulador += krow[2] * irow[j + 0];
			acumulador += krow[3] * irow[j + -1];
			acumulador += krow[4] * irow[j + -2];

			krow = mkernel[2];
			irow = mImage[i];
			acumulador += krow[0] * irow[j + 2];
			acumulador += krow[1] * irow[j + 1];
			acumulador += krow[2] * irow[j + 0];
			acumulador += krow[3] * irow[j + -1];
			acumulador += krow[4] * irow[j + -2];

			krow = mkernel[3];
			irow = mImage[i - 1];
			acumulador += krow[0] * irow[j + 2];
			acumulador += krow[1] * irow[j + 1];
			acumulador += krow[2] * irow[j + 0];
			acumulador += krow[3] * irow[j + -1];
			acumulador += krow[4] * irow[j + -2];

			krow = mkernel[4];
			irow = mImage[i - 2];
			acumulador += krow[0] * irow[j + 2];
			acumulador += krow[1] * irow[j + 1];
			acumulador += krow[2] * irow[j + 0];
			acumulador += krow[3] * irow[j + -1];
			acumulador += krow[4] * irow[j + -2];

            result[(i-li_mImage)*y_length  + (j-li_mImage)] = (uchar) acumulador;
		}
	}

	deleteMemory(mImage, x_mi_length, y_mi_length);

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

bool Filter::deleteMemory(double** &matrix, int x, int y) {
	for (int i = 0; i < x; i++) {
		delete[] matrix[i];
	}

	delete[] matrix;

	return true;
}
