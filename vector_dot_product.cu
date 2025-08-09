#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorDotProduct(const float *A, const float *B, float *partialSums, int n) {
    __shared__ float cache[256];  // shared memory buffer for partial sums per block

    int idx = threadIdx.x + blockDim.x * blockIdx.x; // global thread index
    int cacheIdx = threadIdx.x; // local thread index within block

    float temp = 0.0f;

    // Grid-stride loop: each thread processes multiple elements spaced by total number of threads
    while (idx < n) {
        temp += A[idx] * B[idx];  // accumulate product for this thread
        idx += blockDim.x * gridDim.x;  // jump ahead by total threads to next element
    }

    cache[cacheIdx] = temp; // store thread's partial sum in shared memory
    __syncthreads();       // wait for all threads to write their partial sums

    // Parallel reduction in shared memory:
    // Sum all cache values into cache[0]
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIdx < i) {
            cache[cacheIdx] += cache[cacheIdx + i];  // add pairwise partial sums
        }
        __syncthreads();  // sync before next iteration
        i /= 2;           // half the number of active threads in each step
    }

    // Thread 0 writes the block's total sum to global memory
    if (cacheIdx == 0) {
        partialSums[blockIdx.x] = cache[0];
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
        h_A[i] = 1;
        h_B[i] = 1; 
    }

    // Allocate device memory
    float *d_A, *d_B, *d_partialSums;
    // Allocate device memory for inputs
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    // Allocate device memory for partial sums
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    cudaMalloc(&d_partialSums, blocksPerGrid * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    vectorDotProduct<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_partialSums, n);

    // Copy partial sums back to host
    float *h_partialSums = new float[blocksPerGrid];
    cudaMemcpy(h_partialSums, d_partialSums, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    // Sum partial sums on CPU
    float dot = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) {
        dot += h_partialSums[i];
    }
    std::cout << "Dot product: " << dot << std::endl;

    // Free memory
    delete[] h_partialSums;
    cudaFree(d_partialSums);

    return 0;
}
