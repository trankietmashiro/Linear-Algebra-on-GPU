#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// ===== Matrix Vector Multiplication Kernel =====
// y = Ax
// A is (m x n), x is (n x 1), y is (m x 1)
__global__ void matrixVectorMultiply(const float *A, const float *x, float *y,
                                     int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        float sum = 0.0f;
        for (int col = 0; col < n; col++) {
            sum += A[row * n + col] * x[col];
        }
        y[row] = sum;
    }
}
int main() {
    int m = 4, n = 5; // small example: (4x5) * (5x1) = (4x1)
    size_t sizeA = m * n * sizeof(float);
    size_t sizex = n * sizeof(float);
    size_t sizey = m * sizeof(float);

    float h_MA[20] = { // 4x5
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        10, 11, 12, 13, 14
    };
    float h_Mx[5] = { // 5x1
        1, 
        2, 
        3, 
        4, 
        5
    };
    float h_My[4]; // 4x1 output

    float *d_MA, *d_Mx, *d_My;
    cudaMalloc(&d_MA, sizeA);
    cudaMalloc(&d_Mx, sizex);
    cudaMalloc(&d_My, sizey);

    cudaMemcpy(d_MA, h_MA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Mx, h_Mx, sizex, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;
    matrixVectorMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_MA, d_Mx, d_My, m, n);

    cudaMemcpy(h_My, d_My, sizey, cudaMemcpyDeviceToHost);

    std::cout << "\nMatrix Vector Multiplication Result (4x1):\n";
    for (int i = 0; i < m; i++) {
        std::cout << h_My[i] << "\n";
    }

    cudaFree(d_MA); cudaFree(d_Mx); cudaFree(d_My);

    return 0;
}
