# UHeM HPC Research Report
**Project Number:** 4026552026i  
**Author:** Makbule Özge Özler  
**Date:** April 28, 2026

---

## 0. Workflow & Environment Setup

### Phase 1: Accessing the Cluster
1. **VPN & SSH:** Establish a secure connection to the login node.
   - **Command:** `ssh mozler@altay.uhem.itu.edu.tr`
2. **Resource Allocation:** Request an interactive GPU session via SLURM.
   - **Command:** `srun -p gpu2dq --gres=gpu:1 -n 1 --pty bash`
   - *Note: This allocates 1x NVIDIA A100 GPU on the Altay cluster.*

### Phase 2: Environment Configuration
Inside the compute node (e.g., `a109`), initialize the CUDA toolkit:
- **Load Module:** `module load cuda/cuda-12.5-a100q`
- **Verify:** `nvcc --version` & `nvidia-smi`

### Phase 3: Version Control (GitHub)
To push local changes, use a **Personal Access Token (PAT)** for authentication:
- **User:** `ozgeozler93`
- **Auth:** Use PAT instead of password for `git push`.

---

## 1. System Specifications
| Component | Specification |
| :--- | :--- |
| **Cluster** | Altay (UHeM) |
| **GPU** | NVIDIA A100-PCIE-80GB (Ampere) |
| **CUDA Version** | 12.5 |
| **Profiler** | NVIDIA Nsight Compute (NCU) |

---

## 2. Warm-up Experiments

### Experiment 1: Vector Addition (`vectorAdd.cu`)
- **Objective:** Verify compiler functionality and basic host-to-device memory transfers.
- **Results:** Successful execution (Output: `3.0`).
- **Profiling:** NCU measured DRAM Throughput at ~0.98%.

### Experiment 2: Matrix Multiplication (`gemm.cu`)
- **Objective:** Validate 2D grid/block indexing and global memory coalescing.
- **Results:** Verified $C[0][0] = 64.0$ for $N=32$ (Input: $A=1.0, B=2.0$).
- **Significance:** Foundations for tiled matrix operations, essential for understanding **Voltrix-SpMM**'s 16x8 TCU-blocks.

---

## 3. Profiling Insights
- **Metric Failure:** Attempting to measure `gpu__compute_epipe_tensor_op_utilization` on standard kernels resulted in **Error Code 9**.
- **Conclusion:** Standard CUDA kernels do not utilize Tensor Cores. Future experiments must implement **MMA (Matrix Multiply-Accumulate)** instructions to trigger these units.

## 4. Future Roadmap
- [ ] Implement **Shared Memory Tiling** for GEMM.
- [ ] Analyze **BMat** (Bit-wise Compressed Matrix) format efficiency.
- [ ] Develop a **BMat Decoder** to simulate Voltrix bit-wise compression.
- [ ] Analyze **Warp-Specialization** on A100 architecture.

---
[mozler@a109:~ 468174]$ nano gemm.cu

[mozler@a109:~ 468174]$ nvcc gemm.cu -o gemm

[mozler@a109:~ 468174]$ ./gemm
Matris Carpimi Sonucu [0][0]: 64

---

## 5. Source Codes

<details>
<summary><b>View vectorAdd.cu</b></summary>


```cpp
#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int numElements = 50000;
    size_t size = numElements * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < numElements; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "Result of element 0 (1.0 + 2.0): " << h_C[0] << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}

```

<summary><b>View matrixMul.cu</b></summary>

```cpp
#include <iostream>
#include <cuda_runtime.h>

#define N 32
#define THREADS_PER_BLOCK 16

__global__ void matrixMul(float *A, float *B, float *C, int size) {
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
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N * N; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    matrixMul<<<grid, threads>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    std::cout << "GEMM Result [0][0]: " << h_C[0] << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
```
