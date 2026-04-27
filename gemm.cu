#include <iostream>
#include <cuda_runtime.h>

// Matris boyutu (Sistemi yormamak için 32x32 bir blok seçelim)
#define N 32
#define THREADS_PER_BLOCK 16

__global__ void matrixMul(float *A, float *B, float *C, int size) {
    // 2D Thread ve Blok indekslerini hesapla
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        float sum = 0.0f;
        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

int main() {
    size_t bytes = N * N * sizeof(float);

    // Host (CPU) bellek ayırma
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Matrisleri doldur (Örn: A matrisi 1.0, B matrisi 2.0)
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device (GPU) bellek ayırma
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Veriyi CPU -> GPU taşı
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Grid ve Block boyutlarını ayarla (2D)
    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 grid((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    // Kernel'ı çalıştır
    matrixMul<<<grid, threads>>>(d_A, d_B, d_C, N);

    // Sonucu GPU -> CPU geri çek
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Doğruluk kontrolü (İlk eleman 32*1*2 = 64 olmalı)
    std::cout << "Matris Carpimi Sonucu [0][0]: " << h_C[0] << std::endl;

    // Belleği temizle
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
