# llama-turboquant Quickstart

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with two experimental features:

- **TurboQuant (TQ)** — WHT + Lloyd-Max KV cache quantization at 2/3/4-bit precision for standard attention models
- **KDA State Quantization** — configurable precision for Kimi Delta Attention recurrent state matrices (Kimi-Linear models)

## Branches

| Branch | Description |
|--------|-------------|
| `kda-state-quantization` | Latest — TQ + KDA kernel + state quantization (`-cts` flag) |
| `kda-integration` | TQ + KDA kernel (no state quant CLI) |
| `turboquant-latest` | TQ on latest upstream (no KDA) |
| `turboquant-tq3_0` | Original TQ work |

## Build

### AMD ROCm/HIP

```bash
cmake -B build-hip \
  -DGGML_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMDGPU_TARGETS="gfx1151"   # adjust for your GPU
cmake --build build-hip -j$(nproc)
```

### NVIDIA CUDA

```bash
cmake -B build-cuda \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j$(nproc)
```

## TurboQuant KV Cache Quantization

TurboQuant applies Walsh-Hadamard Transform (WHT) rotation followed by Lloyd-Max optimal quantization to the KV cache at runtime. This reduces KV cache memory by 4-8x with minimal quality loss, based on research from Google's KV cache quantization paper (ICLR 2026).

### How it works

1. **WHT Rotation** — decorrelates the KV cache values within each block of 128 elements, spreading information evenly
2. **Lloyd-Max Codebook** — optimal non-uniform quantization to 2/3/4-bit, minimizing mean squared error
3. **Per-block scale** — each 128-element block stores a single FP16 scale factor

### TQ types

| Type | Bits per weight | Best for |
|------|----------------|----------|
| `tq2_0` | 2.125 bpw | V cache (values are more compressible) |
| `tq3_0` | 3.5 bpw | K cache (keys need more precision for dot products) |
| `tq4_0` | 4.125 bpw | Either K or V when maximum quality is needed |

### Recommended configurations

```bash
# Asymmetric K3/V2 — best memory/quality tradeoff (recommended)
./build-hip/bin/llama-cli \
  -m your-model.gguf \
  -p "Your prompt" \
  -n 128 -ngl 99 \
  -ctk tq3_0 -ctv tq2_0

# Symmetric TQ4 — near-lossless, good for smaller models
./build-hip/bin/llama-cli \
  -m your-model.gguf \
  -p "Your prompt" \
  -n 128 -ngl 99 \
  -ctk tq4_0 -ctv tq4_0

# Symmetric TQ2 — maximum compression (works best at 12B+ model sizes)
./build-hip/bin/llama-cli \
  -m your-model.gguf \
  -p "Your prompt" \
  -n 128 -ngl 99 \
  -ctk tq2_0 -ctv tq2_0
```

### KV cache benchmark results

Tested on AMD Strix Halo (128.5 GB unified memory):

**MoE models (best case for TQ — only 4-10% speed penalty):**

GPT-OSS-120B (128 experts, 4 active, head_dim=64):

| KV Config | Speed (t/s) | Notes |
|-----------|-------------|-------|
| f16/f16 (baseline) | 49.4 | Default |
| tq2_0/tq2_0 | 44.5 | 4-8x KV memory reduction |
| tq4_0/tq4_0 | 39.5 | Near-lossless |
| tq3_0/tq2_0 | 38.9 | Asymmetric, best quality/compression |

**Dense models (higher TQ overhead, 20-27% penalty):**

Qwen3-4B (head_dim=128):

| KV Config | Speed (t/s) | Quality |
|-----------|-------------|---------|
| f16/f16 (baseline) | 44.7 | Reference |
| tq4_0/tq4_0 | 10.0 | Nearly identical to f16 |
| tq3_0/tq3_0 | 10.3 | Coherent |

### Key findings

- **MoE models are the sweet spot** — TQ penalty is minimal because KV cache ops are a smaller fraction of total compute
- **K needs more precision than V** — asymmetric K3/V2 works where symmetric K2/V2 doesn't (especially at smaller model sizes)
- **Minimum model size for TQ2** — symmetric tq2_0/tq2_0 produces coherent output at 12B+, garbled at 4B
- **head_dim must be 128** — models with head_dim=64 or non-standard dims (e.g., MLA's 576/512) are not compatible with TQ block size

## KDA Models (Kimi-Linear)

Kimi-Linear uses KDA (Kimi Delta Attention) — a linear attention mechanism that replaces KV cache with a fixed-size recurrent state matrix. The `-cts` flag controls the precision of this state matrix.

### Download the model

```bash
huggingface-cli download ymcki/Kimi-Linear-48B-A3B-Instruct-GGUF \
  Kimi-Linear-48B-A3B-Instruct-jp-imatrix.Q4_K_M.gguf \
  --local-dir ./models/
```

### Run with optimized state (recommended: F16)

```bash
# F16 state — 46% faster than default F32, no quality loss
./build-hip/bin/llama-cli \
  -m models/Kimi-Linear-48B-A3B-Instruct-jp-imatrix.Q4_K_M.gguf \
  -p "Explain quantum computing" \
  -n 128 -ngl 99 -c 4096 \
  -cts f16
```

### State quantization benchmark results

Tested on AMD Strix Halo (128.5 GB unified memory), Kimi-Linear-48B-A3B Q4_K_M:

| State Type | Flag | Speed (t/s) | Context Mem | Quality |
|-----------|------|-------------|-------------|---------|
| F32 (default) | — | 39.28 | 74 MiB | Good |
| **F16** | `-cts f16` | **57.34** | 54 MiB | **Good** |
| Q8_0 | `-cts q8_0` | 50.42 | 44 MiB | Good |
| TQ3_0 | `-cts tq3_0` | 22.84 | 38 MiB | Marginal |
| TQ4_0 | `-cts tq4_0` | 18.88 | 39 MiB | Broken |
| TQ2_0 | `-cts tq2_0` | 28.17 | 36 MiB | Broken |

**Key finding:** F16 is the sweet spot for KDA state. Sub-8-bit quantization causes error accumulation because the state is read-modify-write every token (unlike KV cache which is write-once).

## OpenAI-Compatible Server (1M Context)

KDA enables massive context windows because the recurrent state is fixed-size (91 MiB regardless of context length). Only the MLA attention layers (every 4th layer) need KV cache.

```bash
# Full 1M token context with F16 state + Q8_0 KV cache for MLA layers
./build-hip/bin/llama-server \
  -m models/Kimi-Linear-48B-A3B-Instruct-jp-imatrix.Q4_K_M.gguf \
  -ngl 99 -c 1048576 \
  -cts f16 -ctk q8_0 -ctv q8_0 \
  --host 0.0.0.0 --port 8080
```

The server exposes OpenAI-compatible endpoints at `http://localhost:8080/v1`:
- `POST /v1/chat/completions`
- `POST /v1/completions`

Use with any OpenAI-compatible client (VS Code extensions, Python openai library, etc.).

## New CLI Flags

| Flag | Description |
|------|-------------|
| `-cts TYPE` / `--cache-type-s TYPE` | Set recurrent state data type for KDA/SSM models. Allowed: `f32`, `f16`, `q8_0`, `tq2_0`, `tq3_0`, `tq4_0`, etc. Default: `f32` |
| `-ctk TYPE` / `--cache-type-k TYPE` | Set K cache type (existing llama.cpp flag, works with TQ types) |
| `-ctv TYPE` / `--cache-type-v TYPE` | Set V cache type (existing llama.cpp flag, works with TQ types) |

## AMD ROCm Note

For Strix Halo (gfx1151) or other unsupported GPUs, set:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.5.1
```
