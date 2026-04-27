# UHeM Baglanti Adimlari



# UHeM Isınma Turları ve GPU Profiling Raporu
**Proje No:** 4026552026i

## 1. Ortam Bilgileri
- **Küme:** Altay
- **Donanım:** NVIDIA A100-PCIE-80GB
- **Yazılım:** CUDA 12.5, NCU (Nsight Compute)

## 2. Deney 1: Vektör Toplama (VectorAdd)
- **Amaç:** CUDA derleme ve temel bellek transferi kontrolü.
- **Sonuç:** Başarıyla çalıştırıldı (1.0 + 2.0 = 3.0).
- **Profiling:** NCU ile DRAM Throughput ölçüldü (%0.98).

## 3. Bulgular
Basit kernel'lar Tensor Core birimlerini tetiklemediği için `gpu__compute_epipe_tensor_op_utilization` metriği raporlanamamıştır.
