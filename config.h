#ifndef CONFIG_H
#define CONFIG_H
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

namespace Config
{
    constexpr int n = 5;
    constexpr int m = 2;
    constexpr int p = 5;
    constexpr int r = 2;
    
    //Allocate, free and synchronize
    inline void allocate(float*& ptr, size_t dim1, size_t dim2) {
        cudaMallocManaged(&ptr, dim1 * dim2 * sizeof(float));
    }    
    inline void free(float* ptr) {
        if (ptr != nullptr){
            cudaFree(ptr);
            ptr = nullptr;
        }
    }    
}

#endif // CONFIG_H