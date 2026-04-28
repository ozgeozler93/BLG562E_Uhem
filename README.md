# UHeM HPC Research Report
**Project No:** 4026552026i

## 0. Connection and Environment Setup
- **Access:** Established via VPN and SSH to `altay.uhem.itu.edu.tr`.

- **Step 1:** VPN & SSH Access
Access the UHeM network via VPN, then connect to the login node:

Command: ssh <user_name>@altay.uhem.itu.edu.tr

Username: mozler

Password: [Your UHeM Password]

- **Step 2:** Requesting GPU Resources (Interactive Session)
To move from the login node to a GPU-enabled compute node:

Command: srun -p gpu2dq --gres=gpu:1 -n 1 --pty bash

Note: This allocates 1 NVIDIA A100 GPU and 1 CPU core under the project utcfsm.

- **Step 3:** Environment Configuration
Once inside the compute node (e.g., a109), load the necessary software stack:

Command: module load cuda/cuda-12.5-a100q

Verification: nvcc --version & nvidia-smi

- **Step 4:** GitHub Integration (Personal Access Token)
When pushing changes to GitHub, use a Personal Access Token (PAT) instead of your password:

Username: ozgeozler93

Password/Token: ghp_XXXXXXXXXXXXXXXXXXXX (Use your generated PAT)

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
