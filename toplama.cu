#include <iostream>
#include <cuda_runtime.h>

// GPU tarafında çalışacak fonksiyon (Kernel)
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x; // Thread'in dünya üzerindeki eşsiz kimliği
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int n = 10000;
    size_t size = n * sizeof(float);

    // 1. CPU belleğinde yer ayır (Host)
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < n; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    // 2. GPU belleğinde yer ayır (Device)
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // 3. Veriyi CPU'dan GPU'ya kopyala
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 4. Kernel'ı çalıştır (Her blokta 256 thread olsun)
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // 5. Sonucu GPU'dan CPU'ya geri çek
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Ilk eleman sonucu (1.0 + 2.0): " << h_C[0] << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
