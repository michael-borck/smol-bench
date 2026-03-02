# Benchmarking Guide

This document covers how to run smol-bench benchmarks, what hardware to use, and how to produce the "bang per bit" analysis that fills a genuine gap in the literature.

## What We're Measuring and Why

Most published benchmarks evaluate full-precision models. A January 2026 paper ("Which Quantization Should I Use?") did a thorough job of benchmarking Llama-3.1-8B across all GGUF quantization levels, but nobody has done the equivalent for the 3-4B class across multiple model families. That's the gap smol-bench fills.

We're running two distinct benchmarks that serve different purposes:

**Benchmark A: "Does quantization change the rankings?"** Take 14 models, quantize each at 8 precision levels, and evaluate on standard tasks. This answers whether full-precision fine-tuning rankings hold after Q4_K_M quantization. Publishable as a standalone table/paper.

**Benchmark B: "What's the real user experience?"** Run Q4_K_M models on actual target hardware (CPU, 8GB RAM) and measure tokens/sec, time-to-first-token, and memory usage alongside quality. This connects quality numbers to deployment reality.

## The Test Matrix

### Models

Start with models that have existing full-precision benchmarks for comparison:

| Model | Parameters | Why Include |
|---|---|---|
| Qwen3-4B-Instruct | 4B | distil labs #1 for fine-tuning |
| Qwen3-1.7B | 1.7B | Smallest viable Qwen; tests scaling |
| Llama 3.2-3B-Instruct | 3.2B | Different architecture; strong baseline |
| Llama 3.2-1B-Instruct | 1B | Tests quantization cliff at small scale |
| Phi-4-Mini (3.8B) | 3.8B | Strong reasoning claims |
| Gemma 3-1B-it | 1B | Different tokenizer |
| Gemma-3-4B-it | 4B | Tests scaling directly against Gemma-3-1B-it |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | Distilled reasoning at micro scale |
| SmolLM2-1.7B | 1.7B | HuggingFace on-device contender |
| Ministral 3B | 3B | Mistral edge-optimized |
| Qwen2.5-Coder-3B | 3B | Domain-specific (coding) baseline |
| Phi-4-Mini-Reasoning | 3.8B | Reasoning distillation comparison |
| DeepSeek-R1-Distill-Qwen-7B | 7B | Does a heavily quantized 7B beat a 4B at Q4_K_M? |
| TinyLlama-1.1B | 1.1B | Community baseline |

### Quantization Levels

For each model, quantize at:

| Quant | Approx bpw | Why Include |
|---|---|---|
| BF16 (baseline) | 16 | Reference point; matches published benchmarks |
| Q8_0 | 8 | Near-lossless baseline |
| Q6_K | 6.6 | High quality, moderate compression |
| Q5_K_M | 5.7 | Often cited as best quality/size balance |
| Q4_K_M | 4.8 | The critical data point for local deployment |
| Q4_0 | 4.0 | Simpler quantization; speed comparison |
| Q3_K_M | 3.4 | Tests where quality collapses |
| Q2_K | 2.6 | Extreme compression; likely broken but worth documenting |

That's 14 models x 8 quant levels = 112 model variants. Each takes 1-3 hours to evaluate depending on hardware, so budget accordingly for the full matrix.

### Tasks

Use the same benchmarks as the Open LLM Leaderboard for direct comparability:

**Standard tasks (via lm-evaluation-harness):**
- MMLU (knowledge)
- HellaSwag (commonsense reasoning)
- GSM8K (math reasoning)
- TruthfulQA (factuality)
- ARC-Challenge (science reasoning)

### Speed and Efficiency Metrics

For each model variant, also record:
- File size on disk (MB)
- Peak RAM usage during inference
- Prompt processing speed (tokens/sec at 512 tokens)
- Generation speed (tokens/sec at 128 tokens)
- Time-to-first-token
- Perplexity on a standard corpus (WikiText-2)

## Tools

### Primary: lm-evaluation-harness

The EleutherAI lm-evaluation-harness is the standard. It's the backend for the HuggingFace Open LLM Leaderboard and directly supports GGUF models.

```bash
# Install
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e ".[hf]"

# Evaluate a GGUF model
# IMPORTANT: Always pass a separate tokenizer to avoid multi-hour hangs
lm_eval --model hf \
  --model_args pretrained=/path/to/gguf_folder,gguf_file=qwen3-4b-q4_k_m.gguf,tokenizer=Qwen/Qwen3-4B-Instruct \
  --tasks hellaswag,mmlu,gsm8k,truthfulqa,arc_challenge \
  --device cuda:0 \
  --batch_size 8 \
  --output_path results/qwen3-4b-q4_k_m/
```

The harness outputs structured JSON with per-task scores, making it straightforward to aggregate results across the full matrix.

### Speed Benchmarks: llama-bench

llama.cpp includes a built-in benchmarking tool that measures prompt processing and generation speed:

```bash
# From llama.cpp build directory
./llama-bench \
  -m /path/to/model.gguf \
  -p 512 \    # prompt length
  -n 128 \    # generation length
  -ngl 0      # 0 for pure CPU, or 99 for full GPU offload
```

Run each model variant twice: once with `-ngl 0` (CPU-only, representing the target deployment) and once with `-ngl 99` (full GPU offload, representing best-case scenario).

### Perplexity: llama-perplexity

```bash
./llama-perplexity -m /path/to/model.gguf -f wikitext-2-raw/wiki.test.raw
```

Perplexity is the fastest sanity check for quantization quality. A sharp perplexity jump between Q4_K_M and Q3_K_M confirms the "quantization cliff" at that model size.

### Quantizing Models

Use llama.cpp's quantize tool or HuggingFace's GGUF-my-repo Space:

```bash
# Local quantization
./llama-quantize /path/to/model-f16.gguf /path/to/model-q4_k_m.gguf Q4_K_M

# Or use HuggingFace's online tool (no local compute needed)
# https://huggingface.co/spaces/ggml-org/gguf-my-repo
```

HuggingFace's GGUF-my-repo Space handles conversion and quantization in the cloud for free. Upload a SafeTensors model, select quantization levels, and it produces GGUF files. This is the easiest path for generating the full quant matrix without tying up local hardware.

## Where to Run

### Option 1: RTX 2060 8GB (recommended primary)

**Why this is actually ideal.** The RTX 2060 has 8GB VRAM, which mirrors the total RAM constraint of typical consumer machines. A 4B model at Q4_K_M (~2.5GB) fits comfortably with room for KV cache.

**What it's good for:**
- GPU-accelerated evaluation through lm-eval-harness (faster than CPU for running the full matrix)
- Speed benchmarks with both GPU offload and CPU-only modes
- Perplexity sweeps
- Realistic VRAM pressure testing

**What it's not good for:**
- BF16 baselines for 4B models (needs ~8GB just for weights, leaving nothing for KV cache). Run these on cloud or use a 3090.

### Option 2: RTX 3090 (if available)

24GB VRAM means BF16 baselines for all models fit easily. Use this for the reference runs if you have access to one.

### Option 3: CPU-only on a Student Laptop

The most representative hardware for actual deployment. Run speed benchmarks (`llama-bench` with `-ngl 0`) on whatever 8GB laptop you can get your hands on. This gives you the ground truth numbers for the target audience.

For quality benchmarks (lm-eval-harness), CPU-only evaluation is slow but works. Budget 4-8 hours per model variant on a modern CPU.

### Option 4: Cloud GPU (Lambda, RunPod, Vast.ai)

Rent an A10 (24GB, ~$0.50/hr) or A100 for a day to blitz through the full matrix. A full evaluation at ~2 hours per variant = ~96 GPU-hours = ~$50-100 on Lambda/Vast.ai.

### Recommended Approach

Run the benchmarks in tiers:

**Tier 1 (do first, minimal hardware):** Perplexity sweep of all variants using llama-perplexity on the RTX 2060. This takes minutes per model and immediately shows where the quantization cliffs are. Produces the core "bang per bit" data.

**Tier 2 (core contribution):** lm-eval-harness on Q4_K_M and BF16 for all models. This is the direct comparison that answers "do full-precision rankings hold at Q4_K_M?"

**Tier 3 (full matrix):** Expand to all 8 quant levels for top models. Shows the full degradation curve.

**Tier 4 (deployment reality):** Speed benchmarks on representative hardware (CPU-only, 8GB RAM). Run llama-bench on the 2060 with `-ngl 0` and on an actual student laptop.

## The "Bang Per Bit" Visualisation

This is the novel chart. Plot performance against cost (size, bits, RAM) to show the efficiency frontier for small quantized models.

### Chart 1: Accuracy vs Model Size (GB)

X-axis: actual file size on disk (GB). Y-axis: composite benchmark score.

Each model appears as a line with points at each quant level. This shows the Pareto frontier: which model+quant combination gives you the most capability per GB.

The interesting finding will be whether the curves cross: does a Q5_K_M 1.7B model ever beat a Q3_K_M 4B model at the same file size? That would directly inform which model to recommend for 4GB machines.

### Chart 2: Accuracy vs Bits Per Weight

X-axis: bits per weight (from 2.6 for Q2_K to 16 for BF16). Y-axis: benchmark score per task.

One line per model, one panel per task. This shows which tasks are most quantization-sensitive and at what precision level each model breaks.

### Chart 3: The "Free Compute" Ratio

X-axis: model size in GB. Y-axis: tokens per second (CPU-only).

Overlay with iso-quality contours. This shows the trade-off space users actually navigate: "I have 8GB of RAM, how much quality can I get and how fast?"

### Chart 4: Quality Recovery Through Fine-Tuning

Once fine-tuned adapters exist (e.g., from [LocoLLM](https://github.com/michael-borck/loco-llm)):

X-axis: quant level. Y-axis: benchmark score. Two lines per model: base and with adapter.

This shows whether fine-tuning recovers quantization losses. If the adapter lines are flatter than the base lines, it proves that task-specific fine-tuning buffers against quantization degradation. That would be a genuinely novel finding.

## Publishing the Results

### As a HuggingFace Dataset

Upload the raw results as a HuggingFace Dataset. This makes the data reproducible and citable.

```
smol-bench/quantized-slm-benchmark
  results/
    qwen3-4b-instruct/
      bf16.json
      q8_0.json
      ...
    llama-3.2-3b-instruct/
      ...
  speed/
    rtx2060_gpu.csv
    rtx2060_cpu.csv
    student_laptop_cpu.csv
  metadata.json
```

### As a HuggingFace Space

Build a Gradio or Streamlit dashboard that:
- Shows the bang-per-bit charts interactively
- Lets users filter by model, quant level, and task
- Links to the raw dataset for reproducibility

### As a Technical Report

The benchmark data frames as:

1. **The gap:** Published benchmarks don't cover quantized 3-4B models systematically
2. **The method:** Standard evaluation applied to a controlled matrix of models x quant levels x tasks x hardware
3. **The findings:** Rankings, quantization cliffs, task sensitivity, efficiency frontiers
4. **The implication:** Which configurations are viable for resource-constrained deployment

## Estimated Time and Cost

| Tier | What | Hardware | Time | Cost |
|---|---|---|---|---|
| 1 | Perplexity sweep (112 variants) | RTX 2060 | ~8 hours | $0 |
| 2 | Core lm-eval (28 variants) | RTX 2060 + cloud | ~48 hours | $0-25 |
| 3 | Full matrix (112 variants) | Cloud A10 | ~96 hours | $50-100 |
| 4 | Speed benchmarks | RTX 2060 + laptop | ~4 hours | $0 |

## Key References

- **lm-evaluation-harness**: https://github.com/EleutherAI/lm-evaluation-harness
- **llama.cpp (quantization + benchmarking)**: https://github.com/ggml-org/llama.cpp
- **HuggingFace GGUF-my-repo**: https://huggingface.co/spaces/ggml-org/gguf-my-repo
- **"Which Quantization Should I Use?" (Llama-3.1-8B study)**: https://arxiv.org/html/2601.14277v1
- **LLM Inference Benchmarking Cheat Sheet**: https://llm-tracker.info/howto/LLM-Inference-Benchmarking-Cheat%E2%80%91Sheet-for-Hardware-Reviewers
