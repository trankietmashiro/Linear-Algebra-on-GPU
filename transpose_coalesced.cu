#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

// ============== Optimized Transpose (Shared Memory) ==============
__global__ void transposeCoalesced(float *out, const float *in, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 avoids bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load to shared memory
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }

    __syncthreads();

    // Transpose coordinates
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Store to output
    if (x < height && y < width) {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    int m = 4, n = 5; // small example: (4x5)
    size_t size = m * n * sizeof(float);

    float h_MA[20] = { // 4x5
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20
    };
    float h_MAT[20]; // 4x5 output

    float *d_MA, *d_MAT;
    cudaMalloc(&d_MA, size);
    cudaMalloc(&d_MAT, size);

    cudaMemcpy(d_MA, h_MA, size, cudaMemcpyHostToDevice);

    // Optimized kernel: block size is TILE_DIM x BLOCK_ROWS
    dim3 blockCo(TILE_DIM, BLOCK_ROWS);
    dim3 gridCo((n + TILE_DIM - 1) / TILE_DIM,
                (m + TILE_DIM - 1) / TILE_DIM);;

    transposeCoalesced<<<gridCo, blockCo>>>(d_MAT, d_MA, n, m);
    cudaMemcpy(h_MAT, d_MAT, size, cudaMemcpyHostToDevice);

    printf("Optimized coalesced transpose:\n");
    for (int r = 0; r < n; r++) {
        for (int c = 0; c < m; c++) {
            printf("%5.1f ", h_MAT[r * m + c]);
        }
        printf("\n");
    }
    printf("\n");

    cudaFree(d_MA); cudaFree(d_MAT);

    return 0;
}
