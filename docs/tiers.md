# VRAM Tier Reference

The organising principle of vram-bench: **models are grouped by what fits in your VRAM budget, not by model family or parameter count.**

Within each tier, every model that fits — at any precision level — competes on quality, speed, and efficiency.

## Size Estimation

GGUF file sizes can be estimated from the parameter count and precision level:

```
File size (GB) ≈ parameters (B) × bits_per_weight / 8 × 1.05
```

VRAM usage during inference is higher than file size due to KV cache and runtime overhead:

```
VRAM needed ≈ file_size × 1.2 + 0.5 GB
```

These are conservative estimates. Actual usage depends on context length, batch size, and framework.

## Tier Definitions

### 4GB Tier — GTX 1050 Ti

**Usable VRAM:** ~3.5 GB (after OS/driver reservation)
**Benchmark GPU:** GTX 1050 Ti (112 GB/s)

| Model | Precision | File Size | VRAM Est. | Fits? |
|---|---|---|---|---|
| SmolLM2-135M | BF16 | 0.3 GB | 0.9 GB | Yes |
| SmolLM2-360M | BF16 | 0.7 GB | 1.3 GB | Yes |
| TinyLlama-1.1B | BF16 | 2.2 GB | 3.1 GB | Tight |
| Gemma 3-1B | BF16 | 2.0 GB | 2.9 GB | Yes |
| Llama 3.2-1B | BF16 | 2.0 GB | 2.9 GB | Yes |
| Qwen3-1.7B | Q4_K_M | 1.1 GB | 1.8 GB | Yes |
| SmolLM2-1.7B | Q4_K_M | 1.1 GB | 1.8 GB | Yes |
| DeepSeek-R1-1.5B | Q4_K_M | 1.0 GB | 1.7 GB | Yes |
| Llama 3.2-3B | Q4_K_M | 2.0 GB | 2.9 GB | Yes |
| Ministral 3B | Q4_K_M | 1.9 GB | 2.8 GB | Yes |
| Qwen3-4B | Q4_K_M | 2.5 GB | 3.5 GB | Tight |
| Phi-4-Mini (3.8B) | Q4_K_M | 2.4 GB | 3.4 GB | Tight |
| Gemma 3-4B | Q4_K_M | 2.5 GB | 3.5 GB | Tight |

**Key comparison:** BF16 Gemma-3-1B (~2.0 GB) vs Q4_K_M Qwen3-4B (~2.5 GB). Both fit. Does the 4× larger model at reduced precision beat the tiny model at full precision?

---

### 6GB Tier — GTX 1060 6GB

**Usable VRAM:** ~5.3 GB
**Benchmark GPU:** GTX 1060 6GB (192 GB/s)

Everything from the 4GB tier, plus:

| Model | Precision | File Size | VRAM Est. | Fits? |
|---|---|---|---|---|
| Qwen3-1.7B | BF16 | 3.4 GB | 4.6 GB | Yes |
| SmolLM2-1.7B | BF16 | 3.4 GB | 4.6 GB | Yes |
| DeepSeek-R1-1.5B | BF16 | 3.0 GB | 4.1 GB | Yes |
| All 3-4B models | Q5_K_M+ | 2.7-3.2 GB | 3.7-4.3 GB | Yes |
| DeepSeek-R1-7B | Q4_K_M | 4.4 GB | 5.8 GB | Tight |
| DeepSeek-R1-7B | Q3_K_M | 3.5 GB | 4.7 GB | Yes |

**Key comparison:** BF16 SmolLM2-1.7B (~3.4 GB) vs Q4_K_M DeepSeek-R1-7B (~4.4 GB). The 7B distilled reasoning model at aggressive quantization vs the 1.7B "overtrained" model at full precision.

---

### 8GB Tier — RTX 2060 Super

**Usable VRAM:** ~7.2 GB
**Benchmark GPU:** RTX 2060 Super (448 GB/s)

Everything from lower tiers, plus:

| Model | Precision | File Size | VRAM Est. | Fits? |
|---|---|---|---|---|
| Llama 3.2-3B | BF16 | 6.4 GB | 8.2 GB | Tight |
| Ministral 3B | BF16 | 6.0 GB | 7.7 GB | Tight |
| Qwen2.5-Coder-3B | BF16 | 6.0 GB | 7.7 GB | Tight |
| All 3-4B models | Q8_0 | 4.0-4.2 GB | 5.3-5.5 GB | Yes |
| DeepSeek-R1-7B | Q5_K_M | 5.2 GB | 6.7 GB | Yes |
| DeepSeek-R1-7B | Q4_K_M | 4.4 GB | 5.8 GB | Yes (comfortable) |

**Key comparison:** BF16 Llama-3.2-3B (~6.4 GB) vs near-lossless Q8_0 Qwen3-4B (~4.2 GB) vs Q5_K_M DeepSeek-R1-7B (~5.2 GB). Three different size classes, three precision levels, same VRAM budget.

---

### 12GB Tier — RTX 3060

**Usable VRAM:** ~11 GB
**Benchmark GPU:** RTX 3060 (360 GB/s)

Everything from lower tiers, plus:

| Model | Precision | File Size | VRAM Est. | Fits? |
|---|---|---|---|---|
| Phi-4-Mini (3.8B) | BF16 | 7.6 GB | 9.6 GB | Yes |
| Gemma 3-4B | BF16 | 8.0 GB | 10.1 GB | Yes |
| Qwen3-4B | BF16 | 8.0 GB | 10.1 GB | Yes |
| DeepSeek-R1-7B | Q8_0 | 7.4 GB | 9.4 GB | Yes |
| DeepSeek-R1-7B | BF16 | 14.0 GB | 17.3 GB | No |

**Key comparison:** BF16 Qwen3-4B (~8.0 GB) vs Q8_0 DeepSeek-R1-7B (~7.4 GB). Full-precision 4B vs near-lossless 7B. Does the extra 3B parameters at Q8_0 beat full precision?

---

### Future: Pooled Tiers (Multi-GPU)

These tiers use Colmena's multiple RTX 2060 Supers with VRAM pooled across cards via llama.cpp tensor parallelism.

| Tier | Configuration | Usable VRAM | Per-Card BW | Interconnect |
|---|---|---|---|---|
| 16GB (pooled) | 2× RTX 2060 Super | ~14 GB | 448 GB/s | PCIe 3.0 |
| 24GB (pooled) | 3× RTX 2060 Super | ~21 GB | 448 GB/s | PCIe 3.0 |

**Why this is interesting:** The RTX 2060 Super's 448 GB/s memory bandwidth exceeds the RTX 3060 (360 GB/s), RTX 4060 (272 GB/s), and RTX 4060 Ti (288 GB/s). Token generation is bandwidth-bound, so pooled 2060 Supers may outperform newer single cards at the same total VRAM — despite the PCIe splitting overhead.

The key experiment is measuring how much throughput the PCIe interconnect costs. See [Multi-GPU Inference](future/multi-gpu.md) for the full proposal.

These tiers will be added once single-card benchmarks are complete and the multi-GPU overhead is quantified.

---

## Which Models at Which Tiers?

Summary view — the maximum parameter count that comfortably fits at each precision level:

| Tier | BF16 (full) | Q8_0 | Q4_K_M | Q2_K |
|---|---|---|---|---|
| 4GB | ≤1B | ≤1.7B | ≤4B | ≤7B |
| 6GB | ≤1.7B | ≤3B | ≤7B | ≤7B+ |
| 8GB | ≤3B | ≤4B | ≤7B | ≤7B+ |
| 12GB | ≤4B | ≤7B | ≤14B | ≤14B+ |

This is the decision matrix vram-bench produces data for. The question is always: **within your VRAM budget, which combination of model size and precision gives the best results?**
