[← Back to README](../README.md)

# Performance

Full stage-by-stage optimization history, batched-decode numbers, and the Ollama/llama.cpp cross-engine comparison. The headline numbers are also summarized on the [main README](../README.md#benchmark-vte-vs-ollama-llamacpp).

## Single-sequence decode (batch=1)

| Stage | Throughput (batch=1) | What changed |
|---|---|---|
| Naive per-thread GEMV | ~18.8 tok/s | Baseline: one thread per output element, no coalescing |
| Coalesced GEMV | ~31.5 tok/s | Threads in a block cooperate on one output neuron, coalesced weight reads |
| In-kernel Q4_K/Q6_K dequantization | ~37.9 tok/s | FFN weights dequantized inline during the GEMV instead of a separate dequant pass ("No-Sync Direct Unpack") |
| QKV Two-Pass Split-K fusion | ~41.0 tok/s | Fills all 32 CUs instead of leaving most idle — see [Architecture: QKV fusion](ARCHITECTURE.md#why-qkv-projection-is-fused-and-why-ffn-fusion-is-off) |
| Vectorized sampler | ~62–70 tok/s | Top-k/top-p/softmax/argsort restricted to the ~50 surviving candidates instead of the full 151936-token vocabulary — see [Bugs found during development](BUGS.md) |
| LM Head captured into the HIP Graph | ~71.7 tok/s | Eliminated the last eager (non-graph) kernel launch in the decode loop — see [Bugs found during development](BUGS.md) |
| Split-KV (Flash-Decoding) attention | ~78–79 tok/s, and *flat* across generation length | Fills far more of the 32 CUs during attention instead of 12; fixes a decline that used to reach ~50.8 tok/s by token 250 — see [Bugs found during development](BUGS.md) |
| Keep-alive pulse removed (`VTE_KEEPALIVE_PULSE_MS=0.0`, default) | **~100 tok/s** | A 2ms/tick safety pulse (added to stop WDDM's DPM from downclocking between bursts) became the largest artificial cost once the tick above got fast enough that DPM never gets an idle window long enough to act — see [Bugs found during development](BUGS.md) |

None of the last four rows touched a GEMV kernel's core math — each came from profiling the actual per-token wall-clock budget (not just "the GPU part") and finding real, measurable non-algorithmic costs: CPU-side dispatch overhead, GPU occupancy left on the table, and finally a fixed-cost safety mechanism whose own justification had quietly expired. The GPU-only decode graph itself measures ~130 tok/s in isolation (via HIP Graph replay timing) — the ~100 tok/s end-to-end figure is now close enough to that ceiling that further gains need actual kernel-level work (GEMV efficiency, WMMA), not more overhead-hunting.

## Batched decode

`generate_batch` (same-length prompts, lockstep) — GPU-only, HIP Graph replay in isolation (no sampler, no logits readback):

| Batch size | Aggregate tok/s | Efficiency vs. ideal linear scaling |
|---|---|---|
| 1 | 110.7 | — |
| 2 | 165.3 | 75% |
| 4 | 204.2 (peak) | 46% |
| 8 | 193.5 (regresses) | 22% |

The batch=8 regression is attributed to Infinity Cache (32MB) thrashing: FFN weights are ~27.5MB per layer (~88% of per-layer traffic), and at batch=8 the working set for weight reuse across the batch no longer fits comfortably alongside everything else competing for the cache. This hasn't been fixed yet — see [Known limitations](LIMITATIONS.md).

Independently re-measured and reconfirmed after a full repository reorganization (200.7 tok/s aggregate / 50.2 tok/s per sequence at batch_size=4, over 100 decode ticks) — these numbers reflect the current state of the code, not a one-off historical run. That said, the ~200 tok/s figure is GPU-only (HIP Graph replay in isolation); the same eager-LM-Head/sampler overhead found in single-sequence decode was also present in batched decode.

**Real end-to-end `generate_batch()` throughput at batch_size=4** (sampler + logits readback included, the number that actually matters for a caller): **127.6 tok/s aggregate** (31.9 tok/s/sequence), up from 118.9 tok/s after capturing the LM Head into the batched HIP Graph (same technique as single-sequence decode, adapted to `[batch_size, vocab_size]` geometry) — a real but modest 7.4% gain, smaller than a naive Amdahl's-law projection suggested. See [Bugs found during development](BUGS.md) for why: most of the eagerly-measured LM Head cost turned out to be genuine GPU work (the model's single largest GEMV, batched 4-wide), not removable dispatch overhead.

## Benchmark: VTE vs. Ollama (llama.cpp)

Both models were benchmarked against Ollama running the *exact same GGUF files on disk* (registered with `ollama create <name> -f Modelfile` using `FROM <absolute-path>`, cross-checked via Ollama's own reported SHA256 to confirm it wasn't silently re-quantizing or re-downloading anything). Same prompt, same `num_predict`/`max_tokens`, `temperature=0` on both sides, decode-only timing (prefill/first-token latency excluded on the VTE side; `eval_count`/`eval_duration` used on the Ollama side, the same decode-only window).

**Current numbers** (after the raw-quantized-weight VRAM-reduction pass described below; prompt: "Write a long, detailed essay about the history of space exploration.", ~700 tokens, `temperature=0`):

| Model | VTE | Ollama (llama.cpp) | VTE / Ollama |
|---|---|---|---|
| Qwen2.5 1.5B (Q4_K_M) | 107.85 tok/s | 110.76 tok/s | 97.4% (tied) |
| Qwen2.5 7B (Q4_K_M) | 35.89 tok/s | 42.71 tok/s | 84.0% |
| Granite 4.1 3B (Q8_0) | **55.65 tok/s** | 50.84 tok/s | **109.5% (VTE faster)** |
| Qwen3.5 2B (Q6_K) | 69.00 tok/s | 77.00 tok/s | 89.6% |

This is a real milestone: Granite now decodes *faster* on VTE than on Ollama/llama.cpp with the identical GGUF file, and Qwen2.5 is within measurement noise of a tie. Both results came from the same change — see below.

### Qwen 2.5 7B (Q4_K_M)

Same methodology, same GGUF file on both sides (registered in Ollama via `ollama create qwen25-7b-ref -f Modelfile` with `FROM "<absolute path>"` — the path needs quoting, this repo's own path has a space in it, `Aetheris Flow`, an unquoted `FROM` silently reads garbage). File size matching (4683074208 bytes both sides) is **not** sufficient verification on its own — a raw fixed-offset byte comparison between the two files showed differences starting at byte 0, which looked alarming until parsing both with `gguf.GGUFReader` (which respects each file's own tensor-offset table rather than assuming identical raw layout) confirmed the actual tensor data is byte-identical; Ollama's ingestion repacks the metadata/header section, shifting every tensor's absolute file offset without changing a single weight value. Verify by comparing tensor `sha256` through `gguf.GGUFReader`, not raw file diffing, in future benchmarks. Prompt: "Write a long, detailed essay about the history of space exploration.", `num_predict`/`max_tokens=700`, `temperature=0`, decode-only timing (first token excluded on both sides).

**35.89 tok/s (VTE) vs. 42.71 tok/s (Ollama) — 84.0%.** The largest model registered so far (28 layers, hidden=3584, ffn=18944 — 2.3x Qwen2.5 1.5B's hidden size), and the widest gap of the four models. Per-category profiling (`VTE_PROFILE=1`, eager/`FallbackExecutor` path — the only one instrumented; absolute ms/tok don't transfer to the HIP-Graph production path due to eliminated per-launch dispatch overhead, but the *proportion* between categories does) confirms the gap lives where expected: ~73% of GPU time in the four large GEMV categories (FFN_Gate_Up 28.0%, QKV_proj 18.4%, FFN_Down 15.9%, AttnOutput 6.9%, LMHead 3.8%) — the same shared GEMV/FFN kernels already tuned for Qwen2.5 1.5B and Granite, not a dispatch-overhead or missing-fusion problem (both already ruled out at this model's smaller siblings).

Effective-bandwidth math makes the size-dependent trend concrete: VTE moves its ~4.5GB of resident weights at ~161 GB/s effective (35.89 tok/s), Ollama at ~192 GB/s (42.71 tok/s) — the same style of comparison at 1.5B lands at 97.4% parity, not 84%, so the relative gap to llama.cpp's hand-tuned GEMV genuinely widens as the matrices get bigger. Leading hypothesis (not yet proven down to the instruction level, would need `rocprof` or equivalent): `gemv_q4k.hip.template`'s per-layer working set roughly doubles at this size (ffn=18944 vs 8960), and RDNA3's 32MB Infinity Cache likely holds a smaller fraction of it in cache across the decode of a single token — the same cache-pressure mechanism already documented for the batch=8 batched-decode regression, just triggered by model width here instead of batch size.

**Accepted as the current number, not a pending TODO** — same posture as the Qwen3.5 tok/s gap above. Closing it further would mean re-tuning the shared GEMV kernel's tiling for larger `K`/`N` dimensions specifically, which risks regressing the already-tuned 1.5B/Granite paths and is a real kernel-engineering project, not a quick fix — deferred until there's a specific reason to prioritize it over other work.

<details>
<summary>Previous numbers (before the VRAM-reduction pass), kept for history</summary>

| Model | VTE | Ollama (llama.cpp) | VTE / Ollama |
|---|---|---|---|
| Qwen2.5 1.5B (Q4_K_M) | 101.68 tok/s (9.84 ms/tok) | 114.67 tok/s (8.72 ms/tok) | 88.7% |
| Granite 4.1 3B (Q8_0) | 45.66 tok/s (21.90 ms/tok) | 51.46 tok/s (19.43 ms/tok) | 88.7% |

Notably consistent at the time — VTE delivered ~89% of llama.cpp's throughput on both models, an identical ratio despite very different architectures (different RoPE convention, different quantization format, ~2x the parameters).
</details>

Ollama was used only as a disposable reference during these benchmarks — the models, their Modelfiles, and the Ollama server process were all removed/stopped afterward; VTE has no runtime or build dependency on it.

It's also worth stating plainly what that number means, because it's the actual point of measuring against Ollama at all: the code doing the dispatching here is **Python**, not C++. Ollama's inference math runs in llama.cpp — hand-tuned C++ GEMV kernels with years of upstream optimization, called through a thin Go server. VTE's Python layer builds the compute graph, resolves kernel arguments, and drives every HIP launch through ctypes — the language usually blamed for making GPU dispatch too slow to compete. Beating llama.cpp's throughput on Granite, and tying it on Qwen2.5, with that much interpreter overhead in the loop is direct evidence for this project's core bet: that CPU-side dispatch overhead and VRAM-bandwidth waste are engineering problems you can profile and remove (HIP Graphs, a sampler restricted to top-k, the LM Head captured into the graph, and now raw-quantized weight routing — see [Bugs found during development](BUGS.md)), not an inherent tax Python has to pay.

### VRAM-reduction pass: routing `attn_q/k/v/output` raw instead of dequantizing to FP16

Both `qwen_mapper.py` and `granite_mapper.py` previously dequantized the four attention projection weights (`attn_q`, `attn_k`, `attn_v`, `attn_output`) from their on-disk quantized format (Q4_K/Q6_K/Q8_0) to FP16 at load time, so they could feed the fused QKV+RoPE kernel (`fused_norm_matmul_rope`/`split_k_qkv_pass2`), which only knows how to read raw `__half*` buffers. This roughly doubles those four matrices' footprint in VRAM relative to the actual bytes on disk.

Granite's GGUF is Q8_0 (2x expansion when dequantized to FP16); attacking `attn_q/k/v` in addition to `attn_output` was accepted as a deliberate bigger-risk/bigger-reward move (~700MB more saved, at the cost of losing the QKV+RoPE fusion for the layers where these weights are now raw). The change:

- `granite_mapper.py::is_raw_q8_0_weight` no longer excludes `attn_q/k/v/attn_output` by name — all Q8_0 tensors are now routed raw (dequantized in-kernel by the GEMV instead of at load time).
- `qwen_mapper.py::is_raw_q4k_weight`/`is_raw_q6k_weight` similarly extended to cover `attn_q/attn_k/attn_output` (Q4_K) and `attn_v` (Q6_K).
- The QKV+RoPE fusion is dynamically disabled per-layer whenever any of these four weights is raw for that layer (the fused kernel has no dequant logic and would otherwise read garbage) — both `fallback_executor.py` and `hip_graph_executor.py` check the raw-weight sets before attempting the fusion.

Measured impact:

- **Granite**: VRAM 4010.7MB → 3448.2MB (weights), tok/s 48.26 → 55.65 (used to trail Ollama at 88.7%, now leads at 109.5%).
- **Qwen2.5**: weights ~986MB → 942.0MB, tok/s ~101.68 → 107.85 (closed the gap from 88.7% to 97.4%).

The counter-intuitive result is that losing the QKV+RoPE fusion (previously assumed to be a pure win — see [Architecture: why QKV is fused](ARCHITECTURE.md#why-qkv-projection-is-fused-and-why-ffn-fusion-is-off)) is more than paid back by the reduced VRAM bandwidth traffic of never writing/reading the FP16-expanded copies of these four matrices — at batch=1 decode, VRAM bandwidth for weight reads dominates over the fixed cost of one extra kernel launch. This only holds for `attn_q/k/v/output` specifically; it does not generalize to arbitrarily fusing less — see "Optimizations tried and rejected" below for cases where fusion removal or kernel-count reduction measured as a net loss or null result.

### Qwen 3.5 2B (hybrid Gated DeltaNet)

Same methodology, same GGUF file on both sides: **~69 tok/s** decode-only on VTE vs. **77 tok/s** on Ollama (llama.cpp) — ~89.6% of Ollama's throughput, measured after the correctness work in [Multi-architecture support: Qwen 3.5](QWEN35.md) landed (that document covers the actual bugs found bringing this architecture up; this section is performance only).

Three kernel-fusion/occupancy experiments were tried afterward to close the remaining gap, each measured with the same protocol (150 tokens, decode-only, excluding the first token/prefill) — all three came back as null results, kept in the code (numerically validated, harmless) rather than reverted, so they aren't blindly re-attempted:

1. **Fusing `a_proj`+`b_proj` into one kernel** (later superseded by the larger fusion below): 67.00 tok/s — within noise of the ~66.78 tok/s baseline.
2. **`causal_conv1d` occupancy tuning** (block size 256→64, exactly one RDNA wavefront, grid 24→96 blocks to use all 32 CUs instead of leaving half idle): 66.66 tok/s — null.
3. **Fusing `qkv_proj`+`z_proj`+`a_proj`+`b_proj` into a single kernel** (`fused_gdn_proj.hip.template`, combining Q6_K and Q8_0 dequant logic in one per-line launch, removing 3 of the ~15 nodes per `linear_attention` layer — 54 nodes total, ~15% of the 393-node graph): 67.26 tok/s — still null.

Per-category GPU profiling (`VTE_PROFILE=1`) explains why: the new Gated DeltaNet kernels already account for only ~17.5% of GPU time in the eager profile, so optimizing them further has little headroom left to recover. The remaining ~56% is the large GEMV/FFN kernels Qwen3.5 *shares* with Qwen2.5/Granite (already tuned there) — closing more of the gap to Ollama would mean touching that shared, already-optimized infrastructure and risking regressions on the other two models, not another isolated Qwen3.5-only change. ~69 tok/s is treated as the real number for this architecture for now, not a pending optimization TODO.

## Portability: dynamic Compute-Unit detection (RDNA2/RDNA3, not just the RX 7600)

Several grid-sizing formulas throughout this project were tuned empirically for the RX 7600's 32 CUs specifically (the only card this was developed and measured on) — comments like "espalha por todos os 32 CUs da RX 7600" litter the codebase for a reason. `HIPRuntime.get_num_cus()` (new) reads the real CU count of whatever GPU is active at runtime, so those formulas scale to any RDNA2/RDNA3 card instead of silently under- or over-subscribing a different one:

- **`vte/core/split_kv_attention.py::_chunk_size_for_cus(num_cus)`**: the Split-KV attention chunk size (fixed `32` before) is now `clamp(round(1024 / num_cus), 8, 64)` — reproduces `32` exactly at `num_cus=32` (zero regression on the reference card, verified byte-for-byte identical output), and scales inversely (more CUs → smaller chunks → more parallel blocks; fewer CUs → larger chunks → less reduce-kernel overhead per mostly-idle block) elsewhere.
- **`causal_conv1d`'s grid** (Qwen3.5 Gated DeltaNet) was *already* hardware-agnostic despite its comment implying otherwise — its block count derives from `conv_dim / 64` (a model-width constant, not a GPU-count constant), so it already produces enough blocks (96, for Qwen3.5) to reasonably occupy any RDNA2/RDNA3 card without needing `num_cus` as an input. Comment corrected; no logic change.
- **QKV Two-Pass Split-K's factor of 32** (`vte/core/fused_qkv_dispatch.py`, `split_k_qkv_pass1/pass2.hip.template`) was *not* made dynamic in this pass — the constant is baked into pointer arithmetic in two kernel templates (not a single `#define`, unlike the Split-KV case) and has a real correctness constraint (`hidden_size` must divide evenly by the split factor) that would need verifying against every supported model's `hidden_size` before any change. Deferred as a deliberate scope decision, not an oversight — the Split-KV fix addresses the same underlying problem (a kernel calibrated only for 32 CUs) with much lower risk.

**A real bug found along the way, worth calling out separately from the CU work itself**: `HIPRuntime`'s hand-maintained `hipDeviceProp_t` struct was silently returning garbage (`38911`) for `multi_processor_count` — see the writeup in [Bugs found during development](BUGS.md) for the full investigation (a struct-layout mismatch with the installed driver, and a genuine RDNA quirk where `hipDeviceGetAttribute`'s multiprocessor-count attribute reports WGPs, not raw CUs — `*2` corrects it). `get_num_cus()` uses the fixed, corrected path.

`VTE_NUM_CUS` env var overrides the detected count for testing (e.g. simulating a different card's CU count on the same physical GPU) without needing the actual hardware.

## Optimizations tried and rejected

Kept disabled rather than removed, so they don't get re-implemented blindly:

- **FFN kernel fusion** (RMSNorm+Gate+Up+SiLU in one launch): measured slower than the unfused version (12.7–13.5 tok/s vs. 18.8 tok/s) — register pressure from holding two accumulators in one loop drops occupancy enough to erase the round-trip savings. Kept behind `VTE_ENABLE_FFN_FUSION` for anyone who wants to re-run the experiment. Full root-cause in [Architecture: Why QKV projection is fused, and why FFN fusion is off](ARCHITECTURE.md#why-qkv-projection-is-fused-and-why-ffn-fusion-is-off).
- **WMMA/Tensor Cores**: rejected early for pure batch=1 GEMV (no benefit at that arithmetic intensity). Worth re-measuring now that batch>1 GEMM is a real, validated code path — not yet done.
