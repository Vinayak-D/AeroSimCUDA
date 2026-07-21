//CUDA Header File for Matrix Operations
#include "config.h"
#ifndef MATRIX_OPERATIONS_CUH
#define MATRIX_OPERATIONS_CUH

//For all operations, assume rows*cols <= 1024, 1 block of threads
//Assume a vector is a matrix with 1 column and n rows

//Matrix addition
__global__ void mxAdd(const float* mat1, const float* mat2, float* result, int rows, int cols);
void matrixAdd(const float* mat1, const float* mat2, float* result, int rows, int cols, cudaStream_t stream);

//Matrix subtraction
__global__ void mxSubtract(const float* mat1, const float* mat2, float* result, int rows, int cols);
void matrixSubtract(const float* mat1, const float* mat2, float* result, int rows, int cols, cudaStream_t stream);

//Matrix transpose
__global__ void mxTranspose(const float* mat, float* result, int rows, int cols);
void matrixTranspose(const float* mat, float* result, int rows, int cols, cudaStream_t stream);

//Matrix multiplication helper
__device__ void matrixMultiplyOnce(const float* mat1, const float* mat2, float* result, int rows1, int cols1, int cols2);

//Matrix multiplication (one pair)
__global__ void mxMultiply(const float* mat1, const float* mat2, float* result, int rows1, int cols1, int cols2);
void matrixMultiply(const float* mat1, const float* mat2, float* result, int rows1, int cols1, int cols2, cudaStream_t stream);

//Matrix scalar multiplication
__global__ void mxScalarMultiply(const float* mat, const float scalar, float* result, int rows, int cols);
void matrixScalarMultiply(const float* mat, const float scalar, float* result, int rows, int cols, cudaStream_t stream);

//Matrix assignment
__global__ void mxAssign(const float* source, float* destination, int rows, int cols);
void matrixAssign(const float* source, float* destination, int rows, int cols, cudaStream_t stream);

//Elementwise max
__global__ void mxElementwiseMax(const float* mat1, const float scalar, float* result, int rows, int cols);
void matrixElementwiseMax(const float* mat1, const float scalar, float* result, int rows, int cols, cudaStream_t stream);


#endif // MATRIX_OPERATIONS_CUH