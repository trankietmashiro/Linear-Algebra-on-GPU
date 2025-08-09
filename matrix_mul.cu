#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// ===== Matrix Multiplication Kernel =====
// C = A * B
// A is (m x k), B is (k x n), C is (m x n)
__global__ void matrixMultiply(const float *A, const float *B, float *C,
                                int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int e = 0; e < k; e++) {
            sum += A[row * k + e] * B[e * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main() {
    int m = 4, k = 3, n = 5; // small example: (4x3) * (3x5) = (4x5)
    size_t sizeA = m * k * sizeof(float);
    size_t sizeB = k * n * sizeof(float);
    size_t sizeC = m * n * sizeof(float);

    float h_MA[12] = { // 4x3
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    };
    float h_MB[15] = { // 3x5
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

    matrixMultiply<<<blocks, threads>>>(d_MA, d_MB, d_MC, m, k, n);

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
