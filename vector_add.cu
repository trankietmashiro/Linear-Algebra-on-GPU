#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int n = 1 << 20; // ~1 million elements
    size_t size = n * sizeof(float);

    // Allocate host memory
    float *h_A = new float[n];
    float *h_B = new float[n];
    float *h_C = new float[n];

    // Initialize data
    for (int i = 0; i < n; i++) {
        h_A[i] = sin(i * 0.001f);      // a sine wave
        h_B[i] = cos(i * 0.001f) * 0.5f; // a scaled cosine wave
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Check result
    for (int i = 0; i < 5; i++) {
    std::cout << "A[" << i << "] = " << h_A[i]
                << ", B[" << i << "] = " << h_B[i]
                << ", C[" << i << "] = " << h_C[i] << "\n";
    }

    // Clean up
    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
