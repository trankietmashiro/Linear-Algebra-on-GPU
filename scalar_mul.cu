#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// ===== Scalar Multiplication Kernel =====
// C = a * B
// A is scalar, B is (m x n), C is (m x n)
__global__ void scalarMultiply(const float a, const float *B, float *C,
                                int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int idx = row * n + col;
        C[idx] = a * B[idx];
    }
}

int main() {
    int m = 4, n = 5; // small example: (4x3) * (3x5) = (4x5)
    size_t size = m * n * sizeof(float);

    float h_Ma = 1;
    float h_MB[20] = { // 3x5
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20
    };
    float h_MC[20]; // 4x5 output

    float *d_MB, *d_MC;
    cudaMalloc(&d_MB, size);
    cudaMalloc(&d_MC, size);

    cudaMemcpy(d_MB, h_MB, size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16); // 16x16 threads per block
    dim3 blocks((n + threads.x - 1) / threads.x,
                (m + threads.y - 1) / threads.y);

    scalarMultiply<<<blocks, threads>>>(h_Ma, d_MB, d_MC, m, n);

    cudaMemcpy(h_MC, d_MC, size, cudaMemcpyDeviceToHost);

    std::cout << "\nScalar Multiplication Result (4x5):\n";
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << h_MC[i * n + j] << "\t";
        }
        std::cout << "\n";
    }

    cudaFree(d_MB); cudaFree(d_MC);

    return 0;
}
