# vram-bench

**I have X GB of VRAM — what's the best model I can run?**

HuggingFace tells you which small model is best at full precision. The Open LLM Leaderboard tells you which large model is best on datacenter GPUs. **vram-bench tells you which model is best for your actual card.**

Most benchmarks compare models under ideal conditions. vram-bench compares everything that fits within a given VRAM budget — full-precision small models, quantized larger models, and everything in between — on real consumer hardware.

A full-precision SmolLM2-1.7B and a Q4_K_M Qwen3-4B both fit in 4GB of VRAM. Which one is actually better? That's the question vram-bench answers.

## How It Works

Models are organised by **VRAM tier**, not by model family or parameter count. Each tier represents a class of consumer GPU:

| VRAM Tier | Benchmark GPU | What Competes |
|---|---|---|
| 4GB | GTX 1050 Ti | BF16 models ≤2B, quantized models ≤4B |
| 6GB | GTX 1060 6GB | BF16 models ≤3B, quantized models ≤7B |
| 8GB | RTX 2060 Super | BF16 models ≤4B, quantized 7B models |
| 12GB | RTX 3060 | BF16 models ≤7B, lightly quantized 7B+ |
| 24GB | RTX 3090 | Reference ceiling |

Within each tier, every model that fits competes on equal footing: quality, speed, and efficiency.

Each tier is benchmarked on the **worst-in-class GPU** for that VRAM level. If it runs here, it runs on your card.

## The Model Matrix

All models use **GGUF format** at every precision level — from lossless BF16 down to aggressive Q2_K quantization. GGUF BF16 preserves the original model precision; it's the same weights in a container format optimised for local inference. This means a BF16 SmolLM2-1.7B and a Q4_K_M Qwen3-4B are evaluated through the same toolchain.

### 16 Models (0.135B to 7B)

| Model | Parameters | BF16 Size | Q4_K_M Size |
|---|---|---|---|
| SmolLM2-135M-Instruct | 0.135B | ~0.3 GB | ~0.1 GB |
| SmolLM2-360M-Instruct | 0.36B | ~0.7 GB | ~0.2 GB |
| Gemma 3-1B-it | 1B | ~2.0 GB | ~0.6 GB |
| Llama 3.2-1B-Instruct | 1B | ~2.0 GB | ~0.6 GB |
| TinyLlama-1.1B | 1.1B | ~2.2 GB | ~0.7 GB |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | ~3.0 GB | ~1.0 GB |
| Qwen3-1.7B | 1.7B | ~3.4 GB | ~1.1 GB |
| SmolLM2-1.7B-Instruct | 1.7B | ~3.4 GB | ~1.1 GB |
| Ministral 3B | 3B | ~6.0 GB | ~1.9 GB |
| Qwen2.5-Coder-3B | 3B | ~6.0 GB | ~1.9 GB |
| Llama 3.2-3B-Instruct | 3.2B | ~6.4 GB | ~2.0 GB |
| Phi-4-Mini (3.8B) | 3.8B | ~7.6 GB | ~2.4 GB |
| Phi-4-Mini-Reasoning | 3.8B | ~7.6 GB | ~2.4 GB |
| Gemma 3-4B-it | 4B | ~8.0 GB | ~2.5 GB |
| Qwen3-4B-Instruct | 4B | ~8.0 GB | ~2.5 GB |
| DeepSeek-R1-Distill-Qwen-7B | 7B | ~14.0 GB | ~4.4 GB |

Within each VRAM tier, every model that fits at any precision level competes. For example, in the 4GB tier: SmolLM2-1.7B at BF16 (~3.4GB) competes against Qwen3-4B at Q4_K_M (~2.5GB).

### Tasks

Standard benchmarks from the Open LLM Leaderboard (via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)):

- **MMLU** — knowledge
- **HellaSwag** — commonsense reasoning
- **GSM8K** — math reasoning
- **TruthfulQA** — factuality
- **ARC-Challenge** — science reasoning

Plus speed and efficiency metrics: tokens/sec, time-to-first-token, peak RAM, perplexity.

## Quick Start

```bash
# Install the evaluation harness
pip install lm-eval[hf]

# Evaluate a full-precision small model
lm_eval --model hf \
  --model_args pretrained=HuggingFaceTB/SmolLM2-1.7B-Instruct \
  --tasks mmlu,gsm8k,hellaswag \
  --device cuda:0 \
  --batch_size auto \
  --output_path results/smollm2-1.7b-bf16/

# Evaluate a quantized model at the same VRAM budget
lm_eval --model hf \
  --model_args pretrained=/path/to/gguf/,gguf_file=qwen3-4b-q4_k_m.gguf,tokenizer=Qwen/Qwen3-4B-Instruct \
  --tasks mmlu,gsm8k,hellaswag \
  --device cuda:0 \
  --batch_size auto \
  --output_path results/qwen3-4b-q4_k_m/
```

That comparison — full-precision 1.7B vs quantized 4B at the same VRAM footprint — is the core of what vram-bench measures.

## Turnkey Benchmarking (Planned)

```bash
pip install vram-bench
vram-bench detect          # reads your GPU, reports VRAM tier
vram-bench run --tier 8gb  # runs the appropriate model set for your tier
vram-bench submit          # packages results for community submission
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/download_models.sh` | Download pre-built GGUF models for the benchmark matrix |
| `scripts/convert_and_quantize.sh` | Convert any HuggingFace model to GGUF and quantize to all levels |
| `scripts/benchmark_quality.py` | Run lm-evaluation-harness across GGUF models |
| `scripts/benchmark_speed.py` | Run llama-bench for speed metrics |
| `scripts/generate_chart_data.py` | Generate interactive chart data from results |

### Producing Your Own GGUFs

If a pre-built GGUF doesn't exist for a model, you can create the full quant matrix from any HuggingFace model:

```bash
# Convert and quantize a HuggingFace model to all precision levels
bash scripts/convert_and_quantize.sh HuggingFaceTB/SmolLM2-1.7B-Instruct

# Just Q4_K_M (fast)
bash scripts/convert_and_quantize.sh google/gemma-3-1b-it --q4-only

# Convert and upload to the vram-bench HuggingFace org
bash scripts/convert_and_quantize.sh HuggingFaceTB/SmolLM2-1.7B-Instruct --upload
```

This requires [llama.cpp](https://github.com/ggml-org/llama.cpp) built locally. The conversion is lossless (BF16 GGUF = original precision), and all quantization levels are produced from that BF16 source.

## Hardware

All benchmarks run on [Colmena](docs/colmena.md), a deliberately constrained 8-GPU rig built around an i3-3220 host. Each VRAM tier is benchmarked on the floor card for that tier — if it runs here, it runs on your card.

| VRAM Tier | GPU | Role |
|---|---|---|
| 4GB | GTX 1050 Ti | Floor of 4GB tier |
| 6GB | GTX 1060 6GB | Floor of 6GB tier |
| 8GB | RTX 2060 Super | Floor of 8GB tier |
| 12GB | RTX 3060 | Floor of 12GB tier |
| 24GB | RTX 3090 | Reference ceiling |

See the [Colmena spec sheet](docs/colmena.md) for full system details and benchmark philosophy.

## Documentation

Full documentation is available at the [vram-bench docs site](https://michael-borck.github.io/vram-bench/), including:

- [Colmena](docs/colmena.md) — benchmark reference machine specs and philosophy
- [Benchmarking Guide](docs/guide.md) — methodology, tools, and how to contribute results
- [Quality Analysis](docs/quality.md) — per-task scores and quantization degradation curves
- [Speed Analysis](docs/speed.md) — generation speed, prompt processing, time-to-first-token
- [Bang per Bit](docs/bang-per-bit.md) — Pareto efficiency frontiers and tradeoffs

## Publishing Plan

- **HuggingFace Org ([vram-bench](https://huggingface.co/vram-bench)):** Central home for models, data, and the dashboard
  - **Model repos:** GGUFs we produce ourselves (filling gaps where no pre-built GGUF exists)
  - **Dataset:** Raw benchmark results (JSON from lm-eval + llama-bench CSVs) for reproducibility
  - **Space:** Interactive dashboard for exploring results by VRAM tier
- **Technical Report:** Methodology and findings
- **Blog Post:** Accessible "bang per bit" analysis for the local LLM community

## Related Projects

- [LocoLLM](https://github.com/michael-borck/loco-llm) — uses vram-bench data to inform base model selection for a routed adapter system
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — the evaluation backend
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — quantization and inference engine

## Future Research

These experiments are tracked for potential spinoff into their own projects:

- **[Perceived Intelligence vs Token Rate](docs/future/perceived-intelligence.md)** — does a faster small model feel smarter than a slower large model?
- **[Multi-GPU Inference on Consumer Hardware](docs/future/multi-gpu.md)** — is pooling VRAM across two cheap cards worth the PCIe penalty?

## Contributing

Contributions welcome, especially:

- Benchmark results from hardware we haven't tested on
- Additional models in the 1-7B range
- Speed benchmarks from student laptops and Chromebooks
- Corrections to methodology or analysis

## License

MIT. See [LICENSE](LICENSE).
