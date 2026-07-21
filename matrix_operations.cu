#include "matrix_operations.cuh"


//Matrix addition
__global__ void mxAdd(const float* mat1, const float* mat2, float* result, int rows, int cols) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < rows && col < cols) {
        //global index for the 1D representation of the 2D matrix
        int index = row * cols + col;
        result[index] = mat1[index] + mat2[index];
    }
}

void matrixAdd(const float* mat1, const float* mat2, float* result, int rows, int cols, cudaStream_t stream) {
    // Launch kernel with 1 block of threads
    dim3 threadsPerBlock(cols, rows);
    mxAdd<<<1, threadsPerBlock, 0, stream>>>(mat1, mat2, result, rows, cols);
}


//Matrix subtraction
__global__ void mxSubtract(const float* mat1, const float* mat2, float* result, int rows, int cols) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < rows && col < cols) {
        //global index for the 1D representation of the 2D matrix
        int index = row * cols + col;
        result[index] = mat1[index] - mat2[index];
    }
}

void matrixSubtract(const float* mat1, const float* mat2, float* result, int rows, int cols, cudaStream_t stream) {
    // Launch kernel with 1 block of threads
    dim3 threadsPerBlock(cols, rows);
    mxSubtract<<<1, threadsPerBlock, 0, stream>>>(mat1, mat2, result, rows, cols);
}


//Matrix transpose
__global__ void mxTranspose(const float* mat, float* result, int rows, int cols) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < rows && col < cols) {
        int indexA = row * cols + col;
        int indexB = cols * col + row;
        result[indexB] = mat[indexA];
    }
}

void matrixTranspose(const float* mat, float* result, int rows, int cols, cudaStream_t stream) {
    // Launch kernel with 1 block of threads
    dim3 threadsPerBlock(cols, rows);
    mxTranspose<<<1, threadsPerBlock, 0, stream>>>(mat, result, rows, cols);
}

//Matrix multiplication helper: only kernels can call __device__ functions
__device__ void matrixMultiplyOnce(const float* mat1, const float* mat2, float* result, int rows1, int cols1, int cols2) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    int idxC = cols2 * row + col;
    if (row < rows1 && col < cols2) {
        float sum = 0.0f;
        for (int i = 0; i < cols1; ++i) {
            int idxA = cols1 * row + i;
            int idxB = cols2 * i + col;
            sum += mat1[idxA] * mat2[idxB];
        }
        result[idxC] = sum;
    }
}

//Matrix multiplication (one pair)
__global__ void mxMultiply(const float* mat1, const float* mat2, float* result, int rows1, int cols1, int cols2) {
    matrixMultiplyOnce(mat1, mat2, result, rows1, cols1, cols2);
}

//Matrix multiplication (one pair)
void matrixMultiply(const float* mat1, const float* mat2, float* result, int rows1, int cols1, int cols2, cudaStream_t stream) {
    // Launch kernel with 1 block of threads
    dim3 threadsPerBlock(cols2, rows1);
    mxMultiply<<<1, threadsPerBlock, 0, stream>>>(mat1, mat2, result, rows1, cols1, cols2);
}

//Matrix scalar multiplication
__global__ void mxScalarMultiply(const float* mat, const float scalar, float* result, int rows, int cols) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < rows && col < cols) {
        int index = row * cols + col;
        result[index] = mat[index] * scalar;
    }
}

void matrixScalarMultiply(const float* mat, const float scalar, float* result, int rows, int cols, cudaStream_t stream) {
    // Launch kernel with 1 block of threads
    dim3 threadsPerBlock(cols, rows);
    mxScalarMultiply<<<1, threadsPerBlock, 0, stream>>>(mat, scalar, result, rows, cols);
}

//Elementwise max
__global__ void mxElementwiseMax(const float* mat1, const float scalar, float* result, int rows, int cols) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < rows && col < cols) {
        int index = row * cols + col;
        result[index] = fmaxf(mat1[index], scalar);
    }
}

void matrixElementwiseMax(const float* mat1, const float scalar, float* result, int rows, int cols, cudaStream_t stream) {
    // Launch kernel with 1 block of threads
    dim3 threadsPerBlock(cols, rows);
    mxElementwiseMax<<<1, threadsPerBlock, 0, stream>>>(mat1, scalar, result, rows, cols);
}

//Matrix assignment
__global__ void mxAssign(const float* source, float* destination, int rows, int cols) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < rows && col < cols) {
        //global index for the 1D representation of the 2D matrix
        int index = row * cols + col;
        destination[index] = source[index];
    }
}

void matrixAssign(const float* source, float* destination, int rows, int cols, cudaStream_t stream) {
    // Launch kernel with 1 block of threads
    dim3 threadsPerBlock(cols, rows);
    mxAssign<<<1, threadsPerBlock, 0, stream>>>(source, destination, rows, cols);
}

