#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// ===== Matrix Multiplication Kernel =====
// C = A + B
// A is (m x n), B is (m x n), C is (m x n)
__global__ void matrixAdd(const float *A, const float *B, float *C,
                                int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int idx = row * n + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int m = 4, n = 5; 
    size_t sizeA = m * n * sizeof(float);
    size_t sizeB = m * n * sizeof(float);
    size_t sizeC = m * n * sizeof(float);

    float h_MA[20] = { // 4x5
        1, 2, 3, 4, 5,
        4, 5, 6, 7, 8,
        7, 8, 9, 10, 11,
        10, 11, 12, 13, 14
    };
    float h_MB[20] = { // 4x5
        3, 2, 1, 5, 4,
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15
    };
    float h_MC[20]; // 4x5 output

    float *d_MA, *d_MB, *d_MC;
    cudaMalloc(&d_MA, sizeA);
    cudaMalloc(&d_MB, sizeB);
    cudaMalloc(&d_MC, sizeC);

    cudaMemcpy(d_MA, h_MA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MB, h_MB, sizeB, cudaMemcpyHostToDevice);

    dim3 threads(16, 16); // 16x16 threads per block
    dim3 blocks((n + threads.x - 1) / threads.x,
                (m + threads.y - 1) / threads.y);

    matrixAdd<<<blocks, threads>>>(d_MA, d_MB, d_MC, m, n);

    cudaMemcpy(h_MC, d_MC, sizeC, cudaMemcpyDeviceToHost);

    std::cout << "\nMatrix Multiplication Result (4x5):\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << h_MC[i * n + j] << "\t";
        }
        std::cout << "\n";
    }

    cudaFree(d_MA); cudaFree(d_MB); cudaFree(d_MC);

    return 0;
}
