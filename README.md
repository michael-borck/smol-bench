# smol-bench

**Systematic benchmarks of quantized small language models on consumer hardware.**

Most published LLM benchmarks evaluate full-precision models on data centre GPUs. smol-bench fills the gap: what actually happens when you quantize 1-7B parameter models to 4-bit and run them on hardware you can buy secondhand for under $200?

## What This Is

A controlled evaluation of **14 models** across **8 quantization levels** (112 variants) on standard benchmarks, measured on real consumer hardware.

The goal is to answer questions that existing leaderboards do not:

- **Do full-precision rankings hold after quantization?** If Model A beats Model B at BF16, does it still win at Q4_K_M?
- **Where are the quantization cliffs?** At what precision level does each model break, and which tasks break first?
- **What is the efficiency frontier?** For a given RAM budget, which model+quant combination gives the most capability per byte?
- **What does deployment actually feel like?** Tokens per second, time-to-first-token, and memory usage on CPU-only hardware.

## The Test Matrix

### Models

| Model | Parameters | Why Include |
|---|---|---|
| Qwen3-4B-Instruct | 4B | distil labs #1 for fine-tuning |
| Qwen3-1.7B | 1.7B | Smallest viable Qwen; tests scaling |
| Llama 3.2-3B-Instruct | 3.2B | Different architecture; strong baseline |
| Llama 3.2-1B-Instruct | 1B | Tests quantization cliff at small scale |
| Phi-4-Mini (3.8B) | 3.8B | Strong reasoning claims |
| Gemma 3-1B-it | 1B | Different tokenizer |
| Gemma 3-4B-it | 4B | Scaling comparison against 1B |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | Distilled reasoning at micro scale |
| SmolLM2-1.7B | 1.7B | HuggingFace on-device contender |
| Ministral 3B | 3B | Mistral edge-optimized |
| Qwen2.5-Coder-3B | 3B | Domain-specific (coding) baseline |
| Phi-4-Mini-Reasoning | 3.8B | Reasoning distillation comparison |
| DeepSeek-R1-Distill-Qwen-7B | 7B | Does a heavily quantized 7B beat a 4B at Q4_K_M? |
| TinyLlama-1.1B | 1.1B | Community baseline |

### Quantization Levels

| Quant | Approx bpw | Purpose |
|---|---|---|
| BF16 | 16 | Reference baseline |
| Q8_0 | 8 | Near-lossless |
| Q6_K | 6.6 | High quality, moderate compression |
| Q5_K_M | 5.7 | Often cited as best quality/size balance |
| Q4_K_M | 4.8 | The critical data point for local deployment |
| Q4_0 | 4.0 | Simpler quantization; speed comparison |
| Q3_K_M | 3.4 | Tests where quality collapses |
| Q2_K | 2.6 | Extreme compression; documenting the floor |

### Tasks

Standard benchmarks from the Open LLM Leaderboard (via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)):

- **MMLU** — knowledge
- **HellaSwag** — commonsense reasoning
- **GSM8K** — math reasoning
- **TruthfulQA** — factuality
- **ARC-Challenge** — science reasoning

Plus speed and efficiency metrics: tokens/sec, time-to-first-token, peak RAM, perplexity.

## Key Finding (Preliminary)

Q4_K_M consistently sits in the efficiency sweet spot — retaining **90-95% of BF16 quality** at roughly **30% of the file size**. Below Q3_K_M, quality collapses sharply for knowledge-heavy tasks.

> **Note:** Current charts show simulated data generated with realistic degradation curves. They will be replaced with real benchmark results as evaluation completes.

## Quick Start

Run a single useful benchmark today:

```bash
# Install the evaluation harness
pip install lm-eval[hf]

# Evaluate BF16 baseline
lm_eval --model hf \
  --model_args pretrained=Qwen/Qwen3-4B-Instruct \
  --tasks mmlu,gsm8k,hellaswag \
  --device cuda:0 \
  --batch_size auto \
  --output_path results/qwen3-4b-bf16/

# Evaluate the same model at Q4_K_M
lm_eval --model hf \
  --model_args pretrained=/path/to/gguf/,gguf_file=qwen3-4b-q4_k_m.gguf,tokenizer=Qwen/Qwen3-4B-Instruct \
  --tasks mmlu,gsm8k,hellaswag \
  --device cuda:0 \
  --batch_size auto \
  --output_path results/qwen3-4b-q4_k_m/
```

That single comparison (BF16 vs Q4_K_M for one model on 3 tasks) takes a few hours on an RTX 2060 and immediately shows how much quantization costs.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/benchmark_quality.py` | Run lm-evaluation-harness across GGUF models |
| `scripts/benchmark_speed.py` | Run llama-bench for speed metrics |
| `scripts/generate_chart_data.py` | Generate interactive chart data from results |

## Hardware

All benchmarks are designed to run on consumer hardware. Reference machines:

- **Primary:** Ryzen desktop + RTX 2060 SUPER (8GB VRAM) — mirrors the 8GB constraint
- **CPU-only:** Any 8GB RAM laptop — the actual deployment target
- **Optional:** RTX 3090 or cloud A10 for BF16 baselines that don't fit in 8GB

If it doesn't run on hardware you can buy secondhand for under $200, it doesn't count.

## Documentation

Full documentation is available at the [smol-bench docs site](https://michael-borck.github.io/smol-bench/), including:

- [Benchmarking Guide](docs/guide.md) — methodology, tools, and how to contribute results
- [Quality Analysis](docs/quality.md) — per-task scores and quantization degradation curves
- [Speed Analysis](docs/speed.md) — generation speed, prompt processing, time-to-first-token
- [Bang per Bit](docs/bang-per-bit.md) — Pareto efficiency frontiers and tradeoffs

## Publishing Plan

- **HuggingFace Dataset:** Raw results (JSON from lm-eval + llama-bench CSVs) for reproducibility
- **HuggingFace Space:** Interactive dashboard for exploring the data
- **Technical Report:** arXiv paper documenting methodology and findings
- **Blog Post:** Accessible "bang per bit" analysis for the local LLM community

## Related Projects

- [LocoLLM](https://github.com/michael-borck/loco-llm) — uses smol-bench data to inform base model selection for a routed adapter system
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — the evaluation backend
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — quantization and inference engine

## Contributing

Contributions welcome, especially:

- Benchmark results from hardware we haven't tested on
- Additional models in the 1-7B range
- Speed benchmarks from student laptops and Chromebooks
- Corrections to methodology or analysis

## License

MIT. See [LICENSE](LICENSE).
