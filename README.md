# UHeM HPC Research Report
**Project No:** 4026552026i

## 0. Connection and Environment Setup
- **Access:** Established via VPN and SSH to `altay.uhem.itu.edu.tr`.
- **Job Scheduling:** Interactive session initiated using SLURM:  
  `srun -p gpu2dq --gres=gpu:1 -n 1 --pty bash`

## 1. System Specifications
- **Cluster:** Altay
- **Hardware:** NVIDIA A100-PCIE-80GB (Ampere Architecture)
- **Software:** CUDA 12.5, NCU (Nsight Compute)

## 2. Warm-up Experiments (2026-04-27)

### Experiment 1: Vector Addition (vectorAdd.cu)
- **Objective:** Verify CUDA compiler (nvcc) functionality and basic HtoD/DtoH memory transfers.
- **Results:** Executed successfully with correct output (1.0 + 2.0 = 3.0).
- **Profiling:** Measured using `ncu`. DRAM Throughput reached ~0.98% of peak for this minimal workload.

### Experiment 2: Matrix Multiplication (gemm.cu)
- **Objective:** Validating 2D grid/block indexing and global memory access.
- **Results:** Output verified ($C[0][0] = 64.0$ for $N=32$ with $A=1.0, B=2.0$). 
- **Significance:** Confirmed the foundational logic for tiled matrix operations, a prerequisite for understanding the specialized 16x8 TCU-blocks in Voltrix-SpMM.

---
[mozler@a109:~ 468174]$ nano gemm.cu

[mozler@a109:~ 468174]$ nvcc gemm.cu -o gemm

[mozler@a109:~ 468174]$ ./gemm
Matris Carpimi Sonucu [0][0]: 64

---

## 3. Profiling Insights
- Standard CUDA kernels do not trigger Tensor Core units.
- Attempting to measure `gpu__compute_epipe_tensor_op_utilization` on non-MMA kernels resulted in error code 9, confirming that specialized Tensor Core instructions (like those in Voltrix-SpMM) are required for these metrics.

## 4. Next Steps: Voltrix-SpMM Integration
- Analyze **BMat** (Bit-wise Compressed Matrix) format efficiency.
- Implement warp-specialized producer-consumer models on A100 nodes.
