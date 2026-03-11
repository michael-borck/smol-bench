# Multi-GPU AI Inference: A Practical Guide
### Local LLM Hardware, VRAM Pooling, and the Cost-Effectiveness Question

---

## 1. Why Multiple GPUs for Local LLM Inference?

The single biggest constraint in running large language models locally is **VRAM** (Video RAM). A model's minimum VRAM requirement is roughly:

> `model parameters × bytes per parameter`

So a 7B parameter model at 4-bit quantization needs roughly **4 GB**, a 13B model needs ~7–8 GB, a 34B model needs ~20 GB, and a 70B model needs ~40 GB. Most consumer GPUs top out at 8–24 GB, which puts anything above 13B out of reach on a single card.

Multi-GPU setups solve this in one of two ways:

- **Running bigger models** than any single card could hold (VRAM pooling)
- **Running faster inference** by parallelising the compute across cards (though gains are more nuanced — see Section 3)

For most home lab and research users, the first goal is the dominant reason to go multi-GPU.

---

## 2. How Inference Engines Split Models Across GPUs

The UI layer (Open WebUI, LM Studio, etc.) does not do the heavy lifting. The inference *engine* running underneath manages the GPU splits. There are two main strategies:

### 2.1 Pipeline Parallelism (Layer Offloading)

Used by: **llama.cpp**, **Ollama** (via llama.cpp under the hood), **LM Studio**, **Jan**

The model's layers are divided sequentially across GPUs like an assembly line:

```
GPU 0: layers 1–12   →   GPU 1: layers 13–24   →   GPU 2: layers 25–36
```

Each token generated travels down the pipeline from card to card via your PCIe bus. Only one GPU is active at a time per token pass.

**Implications:**
- Identical GPUs: each stage takes the same time, no bottleneck between stages
- Mixed GPUs: total throughput is limited by the slowest card's bandwidth
- PCIe slot speed matters — a card in a x4 slot starves even if the card itself is fast
- Overall throughput does **not** scale linearly with GPU count; you're running a larger model, not a faster one

### 2.2 Tensor Parallelism (Shared Math)

Used by: **ExLlamaV2**, **vLLM**, **TensorRT-LLM**

Every layer is sliced in half (or thirds, etc.) and each GPU computes its slice simultaneously:

```
Layer N: GPU 0 computes left half | GPU 1 computes right half — simultaneously
         ↓ synchronise results ↓
Layer N+1: repeat
```

**Implications:**
- GPUs work in parallel — genuine throughput gains are possible
- Synchronisation overhead between cards adds latency — requires fast interconnects (NVLink ideal, PCIe acceptable for same-generation cards)
- Mixed-speed GPUs still bottleneck at the slower card per synchronisation step
- Higher tokens/sec ceiling than pipeline parallelism on matched hardware

### 2.3 Which to Use?

| Scenario | Recommended Approach |
|---|---|
| Just need the model to fit in VRAM | Pipeline parallelism (Ollama/llama.cpp) — zero config |
| Matched GPUs, want maximum t/s | Tensor parallelism via ExLlamaV2 |
| Mixed GPU generations | Pipeline parallelism — tensor parallelism's sync overhead punishes mixed speeds more |
| Fine-tuning | Neither; use a single fast GPU with bf16 support |

---

## 3. The RTX 2060 Super: An Underrated Inference Card

The RTX 2060 Super is frequently overlooked in LLM discussions because it is an older generation. This is a mistake for inference workloads.

### 3.1 Memory Bandwidth Comparison

LLM inference is a **memory-bandwidth-bound** task. The rate at which weights can be streamed from VRAM into the compute units determines tokens-per-second far more than raw shader throughput.

| GPU | VRAM | Memory Bus | Bandwidth | Gen |
|---|---|---|---|---|
| RTX 2060 Super | 8 GB | 256-bit | **448 GB/s** | Turing |
| RTX 3060 12 GB | 12 GB | 192-bit | 360 GB/s | Ampere |
| RTX 3060 Ti | 8 GB | 256-bit | 448 GB/s | Ampere |
| RTX 3070 | 8 GB | 256-bit | 448 GB/s | Ampere |
| RTX 4060 | 8 GB | 128-bit | 272 GB/s | Ada |
| RTX 4060 Ti 16 GB | 16 GB | 128-bit | 288 GB/s | Ada |
| RTX 4070 | 12 GB | 192-bit | 504 GB/s | Ada |
| RTX 3090 | 24 GB | 384-bit | 936 GB/s | Ampere |
| RTX 4090 | 24 GB | 384-bit | 1,008 GB/s | Ada |

The 2060 Super's 256-bit bus gives it bandwidth parity with the 3060 Ti and 3070 — cards that were released years later at much higher prices. The 4060 and 4060 Ti are significantly *slower* for inference despite being newer.

### 3.2 Three RTX 2060 Supers: What You Actually Get

| Metric | Value |
|---|---|
| Pooled VRAM | 24 GB |
| Per-card bandwidth | 448 GB/s |
| Effective pipeline throughput | ~448 GB/s (sequential) |
| Model size ceiling | ~20–22 GB usable (leaving headroom for KV cache) |
| Comparable single-card VRAM | RTX 3090 / RTX 4090 |

With 3x matched cards, the "weakest link" penalty disappears entirely — every stage in the pipeline runs at the same rate.

### 3.3 Expected Token Rates (Estimates)

These are approximate single-user inference figures for pipeline parallelism:

| Model | Quantisation | VRAM Used | Estimated t/s |
|---|---|---|---|
| Llama 3 8B | Q4_K_M | ~5 GB | 35–50 t/s |
| Mistral 7B | Q4_K_M | ~4.5 GB | 35–55 t/s |
| Qwen 2.5 14B | Q4_K_M | ~9 GB | 20–30 t/s |
| Llama 3 70B | Q2_K | ~26 GB | ❌ too large |
| Llama 3 70B | IQ1_M | ~18 GB | 8–12 t/s |
| Mistral 22B | Q4_K_M | ~13 GB | 14–20 t/s |
| Command-R 35B | Q3_K_M | ~15 GB | 12–18 t/s |
| Yi-34B | Q4_K_M | ~20 GB | 10–15 t/s |

> **Verdict:** For a single-user research or teaching context, 10–20 t/s on a high-quality 34B model is entirely usable. Conversational feel begins around 8 t/s; anything above 15 t/s is comfortable.

### 3.4 Scaling Up: The 6x GPU Chassis Option

A mining-style chassis (e.g., WEIHO 6-GPU) with six RTX 2060 Supers creates an interesting decision point:

| Configuration | Cost (AUD, approx.) | Total VRAM | Architecture |
|---|---|---|---|
| 6x RTX 2060 Super | $600–700 secondhand | 48 GB | 6 independent 8 GB cards |
| WEIHO chassis + PSU | ~$190 | — | PCIe x1 risers |
| **Total** | **~$800–900** | **48 GB** | — |

This is less than a single RTX 3090 (24 GB, ~$800–1000 AUD). But the 48 GB figure is misleading without understanding two very different use modes:

**Pooled VRAM (one large model split across cards):** Using llama.cpp `--tensor-split 1,1,1,1,1,1`, a single ~40 GB model can theoretically run across all six cards. However, mining chassis use **PCIe x1 riser cables**, which dramatically limit inter-GPU bandwidth compared to x8 or x16 slots. This makes layer splitting functional but slow — each pipeline stage waits on the narrow PCIe x1 link to pass activations to the next card. For a single large model, a single RTX 3090 at 936 GB/s bandwidth will be significantly faster than six 2060 Supers connected by risers.

**Independent workers (six separate models):** The compelling use case is running **six independent 7B model instances simultaneously** — one per card, no cross-GPU communication needed. This is ideal for:

- Load balancing across multiple concurrent users
- Mixture-of-Agents (MoA) architectures where multiple models contribute to a response
- Batch evaluation (e.g., running vram-bench across six models in parallel)
- A/B testing different models or quantisation levels side by side

Six concurrent 7B inference workers for under $1000 AUD is remarkable value. The PCIe x1 riser limitation is irrelevant when each card runs independently.

> **Key insight:** The value of many cheap GPUs depends entirely on whether you need one big model or many small ones. For pooled VRAM, fewer cards in proper PCIe slots wins. For concurrent independent inference, more cards wins regardless of slot bandwidth.

### 3.5 Weaknesses to Know

- **No native bf16 support** — the Turing architecture pre-dates bfloat16. This effectively rules out fine-tuning modern LLMs on these cards (QLoRA via fp16 is possible but slower and less stable).
- **Older CUDA cores** — raw TFLOP compute is well behind Ampere/Ada for tasks that are compute-bound (batch inference, training).
- **Thermal management** — three or more cards running sustained inference in a single chassis needs airflow planning. Mining chassis have open-air designs that help, but power draw scales linearly (~75W per card under load).
- **PCIe riser bandwidth** — mining-style x1 risers work fine for independent inference but are a severe bottleneck for layer-split or tensor-split workloads. For pooled VRAM, cards in proper x8/x16 motherboard slots are strongly preferred.

---

## 4. Tooling Overview for Multi-GPU Setups

### 4.1 CLI and Background Engines

| Tool | Multi-GPU | Config Effort | Format Support | Best For |
|---|---|---|---|---|
| **Ollama** | Layer splitting via llama.cpp | Low | GGUF | Quickest path to working multi-GPU chat |
| **llama.cpp CLI** | `--tensor-split` flags | Low (script it) | GGUF | Custom launch scripts, fine-grained layer control |
| **ExLlamaV2** | Explicit per-GPU allocation | Medium | EXL2, GPTQ | Maximum t/s on matched Nvidia cards |
| **TabbyAPI** | ExLlamaV2 backend | Medium | EXL2, GPTQ | Lighter alternative to vLLM for local multi-GPU serving |
| **vLLM** | Tensor parallelism | Medium-High | GPTQ, AWQ, fp16 | High-throughput production serving, batch inference |

### 4.2 UI Frontends

| UI | Deployment | Multi-GPU | Notes |
|---|---|---|---|
| **Open WebUI** | Docker / Python | Via Ollama backend | Best self-hosted ChatGPT clone |
| **LM Studio** | Desktop app | Built-in GPU toggle panel | Most beginner-friendly; visual GPU allocation |
| **Text Gen WebUI** (Oobabooga) | Python / local web | Per-GPU VRAM allocation | Most granular control; power-user tool |
| **AnythingLLM** | Desktop or Docker | Via Ollama/LM Studio | Best for RAG / document Q&A |
| **Jan** | Desktop app | Automatic (llama.cpp) | Clean open-source LM Studio alternative |

### 4.3 Recommended Stack for 3x 2060 Super

**For CLI chat (lowest friction):**
```bash
# Ollama splits layers across GPUs via llama.cpp under the hood.
# Set CUDA_VISIBLE_DEVICES to select GPUs, or let it auto-detect.
CUDA_VISIBLE_DEVICES=0,1,2 ollama run command-r   # ~15GB Q4 model, splits across 3 cards

# For finer control over the split:
# OLLAMA_GPU_SPLIT=8,8,8 ollama run command-r

# Or let Ollama distribute layers automatically:
# Use PARAMETER num_gpu 999 in a Modelfile to push all layers to GPU
```

> **Note:** Ollama's multi-GPU is layer splitting (pipeline parallelism), not true tensor parallelism. It is functionally VRAM pooling — a model larger than one card's VRAM can run across multiple cards — but with PCIe bandwidth overhead between stages. For benchmarking where you need precise control over the split, use llama.cpp directly.

**For maximum performance (ExLlamaV2):**
```python
from exllamav2 import ExLlamaV2, ExLlamaV2Config

config = ExLlamaV2Config()
config.model_dir = "/models/mistral-22b-exl2"
config.prepare()

model = ExLlamaV2(config)
# Explicit split: 8GB each across 3 GPUs
model.load([8192, 8192, 8192])
```

**For a tuned launch script (llama.cpp):**
```bash
./llama-cli \
  -m /models/command-r-q4_k_m.gguf \
  -ngl 99 \                    # push all layers to GPU
  --tensor-split 1,1,1 \       # equal split across 3 cards
  -c 8192 \                    # context window
  --temp 0.7 \
  -i                           # interactive / chat mode
```

**For production-style multi-GPU serving (vLLM):**
```bash
# Genuine tensor parallelism — GPUs work in parallel, not sequentially
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-22B \
  --tensor-parallel-size 3 \
  --gpu-memory-utilization 0.9
```

vLLM provides an OpenAI-compatible API, so any frontend (Open WebUI, etc.) can connect to it. Higher setup complexity than Ollama but better multi-GPU utilisation for a single large model.

**For lightweight local serving (TabbyAPI):**

TabbyAPI uses ExLlamaV2 under the hood with an OpenAI-compatible API. Lighter than vLLM, good multi-GPU support, designed for single-user local inference. A middle ground between Ollama's simplicity and vLLM's production features.

> **Practical note:** These tools don't conflict — you can have Ollama for quick single-card use, llama.cpp for benchmarking with precise control, and vLLM for multi-GPU serving, all on the same machine.

---

## 5. The Apple Silicon Comparison

Apple's M-series chips take a fundamentally different architectural approach. Rather than discrete VRAM on a GPU card, they use **unified memory** — a single high-bandwidth memory pool shared between CPU, GPU, and Neural Engine. This has significant implications for LLM inference.

### 5.1 Apple Silicon Memory Bandwidth

| Chip | Config | Unified RAM | Bandwidth |
|---|---|---|---|
| M4 (base) | Mac Mini entry | 16–32 GB | 120 GB/s |
| M4 Pro | Mac Mini Pro, MacBook Pro | 24–64 GB | 273 GB/s |
| M4 Max | Mac Studio, MacBook Pro | 36–128 GB | 546 GB/s |
| M4 Ultra | Mac Studio top | 192 GB | 819 GB/s |
| M5 Pro *(est.)* | Mac Mini/MacBook Pro | 24–64 GB | ~350 GB/s |
| M5 Max *(est.)* | Mac Studio | 64–128 GB | ~660 GB/s |

> Note: M5 specifications are provisional based on Apple's typical generation-over-generation gains (~20–25% bandwidth improvement). Verify current Apple specs before purchasing.

### 5.2 Single Machine vs. Multiple Macs

A critical point often misunderstood: **multiple Mac Minis do not natively pool their unified memory** the way multi-GPU does on a single PC.

Each Mac Mini is a fully independent computer. To run a model split across two or more Mac Minis, you need distributed inference via tools like:

- **llama.cpp distributed mode** (`--rpc` server mode, experimental)
- **Petals** (distributed transformer inference over a local network)
- **Exo** (open-source project specifically for Apple Silicon cluster inference)

This introduces network latency between nodes (even on 10GbE), which adds overhead that unified-memory solutions do not have. In practice, a Mac Mini cluster for inference is more complex to operate than a multi-GPU PC, and the per-node memory is not additive without latency penalties.

### 5.3 Side-by-Side: 3x RTX 2060 Super vs. Apple Options

| Configuration | Approx. AUD Cost | Effective "VRAM" | Bandwidth (inference) | Notes |
|---|---|---|---|---|
| 3x RTX 2060 Super (secondhand) | $450–700 | 24 GB (pooled) | ~448 GB/s (pipeline) | Needs host PC; power draw ~450W under load |
| Mac Mini M4 (32 GB) | ~$1,800 | 32 GB | 120 GB/s | Quiet, low power (~25W idle), but slow bandwidth |
| Mac Mini M4 Pro (48 GB) | ~$2,300 | 48 GB | 273 GB/s | Better balance; good for 34B models |
| Mac Mini M4 Pro (64 GB) | ~$2,700 | 64 GB | 273 GB/s | Comfortable 70B headroom |
| 2x Mac Mini M4 Pro (48 GB each) | ~$4,600 | 48 GB each (not pooled) | 273 GB/s per node | Distributed only; complex setup |
| Mac Studio M4 Max (128 GB) | ~$4,500+ | 128 GB | 546 GB/s | Serious single-machine performance |
| Mac Studio M4 Ultra (192 GB) | ~$8,000+ | 192 GB | 819 GB/s | Near-datacenter for 70B+ unquantised |

### 5.4 Token Rate Comparison: 34B Model (Q4)

| Platform | Approx. t/s (34B Q4) | Notes |
|---|---|---|
| 3x RTX 2060 Super | 10–15 t/s | Pipeline parallelism; conversational |
| Mac Mini M4 (32 GB) | 8–12 t/s | Bandwidth-limited; fits but slow |
| Mac Mini M4 Pro (48 GB) | 18–25 t/s | Better bandwidth; model fits cleanly |
| Mac Studio M4 Max (128 GB) | 35–55 t/s | Comfortable headroom; fast |

At the 34B model size, the 3x 2060 Super setup is genuinely competitive with a Mac Mini M4 Pro configuration costing 3–4x more — and faster than a base Mac Mini M4.

---

## 6. Cost-Effectiveness Analysis

### 6.1 Bang-Per-Dollar at Common Price Points

Assuming secondhand Australian pricing:

| Config | AUD Cost | Pooled VRAM | Max Model Size | Inference Speed (est.) |
|---|---|---|---|---|
| 2x RTX 2060 Super | $300–450 | 16 GB | ~13B Q4 comfortably | 25–40 t/s on 13B |
| 3x RTX 2060 Super | $450–700 | 24 GB | ~34B Q4 | 10–15 t/s on 34B |
| 4x RTX 2060 Super | $600–950 | 32 GB | ~34B Q4 or 70B IQ2 | 10–14 t/s on 34B |
| RTX 3090 (single) | $800–1,000 | 24 GB | ~34B Q4 | 25–35 t/s on 34B |
| Mac Mini M4 Pro 48 GB | ~$2,300 | 48 GB | ~70B Q4 | 18–25 t/s on 34B |

The **3x RTX 2060 Super** configuration is the most cost-effective entry point into 34B-class model inference. Its main trade-off against a single RTX 3090 of the same VRAM capacity is throughput: the 3090's 936 GB/s bandwidth (vs. the pipeline-limited ~448 GB/s effective rate of the tri-card setup) roughly doubles tokens per second on models that fit entirely on the 3090.

A **4th 2060 Super** adds 8 GB of headroom (32 GB total) for a marginal cost, allowing more breathing room for KV cache on long contexts or slightly less aggressive quantisation on 34B models. The throughput gain is minimal since pipeline parallelism adds another stage.

### 6.2 When the Mac Ecosystem Makes More Sense

- You need **more than 32 GB in a single logical device** without distributed inference complexity
- You prioritise **power efficiency** (a Mac Mini M4 Pro idles at ~8W, a tri-GPU PC at ~150W baseline)
- You want **macOS and the Apple ecosystem** for development
- You are running **70B+ models regularly** — this is where Mac Studio M4 Max/Ultra pulls significantly ahead
- You need **silence** — a three-GPU workstation is not quiet

### 6.3 When the Multi-GPU PC Wins

- **Budget is the primary constraint** — $600 vs. $2,300+ for similar 34B performance
- You already have a **compatible host machine** (the GPU cost dominates)
- You want **flexibility** — swap cards, add more, run CUDA-specific tools
- You are comfortable with **Linux and CLI tooling**
- You plan to **experiment with fine-tuning pipelines** eventually (you would add a bf16-capable card for that role)

---

## 7. Practical Recommendations

### For the LocoLLM Lab / Research Context

Given the existing 3x RTX 2060 Super inventory plus a research focus on cost-effective inference:

1. **Run Ollama as the primary inference backend** — zero-config multi-GPU pooling, REST API available for all frontends and custom scripts.

2. **Add ExLlamaV2 as a secondary path** for benchmarking — EXL2 quantised models at 3.0–4.0 bpw on 34B give a meaningful t/s improvement over GGUF Q4 at the same VRAM footprint.

3. **Target the 14B–22B sweet spot** rather than the absolute 34B ceiling. A Mistral 22B or Qwen2.5 14B at Q6_K will outperform a Yi-34B at Q3 in both quality and speed given your VRAM headroom.

4. **A fourth 2060 Super** is a worthwhile addition if available cheaply — 32 GB enables running Llama 3 70B at IQ2_XS (~17 GB) with a decent context window.

5. **PCIe slot planning matters** — ensure all three (or four) cards are in x8 or x16 electrical slots. A card in a x4 slot will bottleneck inter-GPU data transfer under pipeline parallelism, effectively wasting bandwidth.

6. **For benchmarking smol-bench** — the 2060 Super's profile (high bandwidth, modest compute, older CUDA) makes it a genuinely interesting reference point for research on cost-optimised inference hardware. It represents a class of GPU that is abundant, cheap, and frequently overlooked in the literature.

---

## 8. Summary

Multi-GPU inference is primarily a **VRAM expansion strategy**, not a speed multiplication strategy. Pipeline parallelism (the default in Ollama and llama.cpp) lets you run models that would not otherwise fit, at roughly the throughput of a single card. Tensor parallelism (ExLlamaV2, vLLM) on matched hardware can deliver genuine speed gains but requires more configuration.

The RTX 2060 Super is an excellent inference card hiding in plain sight — its 256-bit bus and 448 GB/s bandwidth outperform most mid-tier Ampere and Ada cards for single-user LLM workloads. Three of them for under $700 AUD gives 24 GB of pooled VRAM and usable performance on 34B-class models.

Apple Silicon's unified memory architecture offers a genuinely different value proposition: simpler setup, better per-GB quality at higher VRAM tiers, lower power consumption, but at a significant cost premium. For a research lab budget, the multi-GPU PC approach delivers strong capability per dollar — particularly if the host machine already exists.

The Mac Mini cluster idea ("stacking" M4 or M5 Minis) is often discussed but rarely practical for inference: it requires distributed inference software, introduces network latency between nodes, and the memory pools are not additive in the same seamless way as multi-GPU on a single PCIe bus. A single well-specced Mac Studio M4 Max is a better investment than two Mac Minis for inference purposes.

---

*Report compiled March 2026. GPU pricing reflects approximate AUD secondhand market. Token rate estimates are single-user, single-stream inference benchmarks and will vary by quantisation, context length, prompt complexity, and driver/software version.*
