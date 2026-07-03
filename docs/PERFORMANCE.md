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

Both models were benchmarked against Ollama running the *exact same GGUF files on disk* (registered with `ollama create <name> -f Modelfile` using `FROM <absolute-path>`, cross-checked via Ollama's own reported SHA256 to confirm it wasn't silently re-quantizing or re-downloading anything). Same prompt, same `num_predict`/`max_tokens` (200), `temperature=0` on both sides, decode-only timing (prefill/first-token latency excluded on the VTE side; `eval_count`/`eval_duration` used on the Ollama side, the same decode-only window):

| Model | VTE | Ollama (llama.cpp) | VTE / Ollama |
|---|---|---|---|
| Qwen2.5 1.5B (Q4_K_M) | 101.68 tok/s (9.84 ms/tok) | 114.67 tok/s (8.72 ms/tok) | 88.7% |
| Granite 4.1 3B (Q8_0) | 45.66 tok/s (21.90 ms/tok) | 51.46 tok/s (19.43 ms/tok) | 88.7% |

Notably consistent — VTE delivers ~89% of llama.cpp's throughput on both models, an identical ratio despite very different architectures (different RoPE convention, different quantization format, ~2x the parameters). That's a reasonable proxy for "the remaining gap is systemic dispatch/kernel-efficiency overhead across the whole engine, not an architecture-specific bug hiding in one model" — llama.cpp has years of GEMV-kernel tuning VTE doesn't have yet, and closing that gap further is a kernel-level efficiency project, not a correctness one. Ollama was used only as a disposable reference during this benchmark — the model, its Modelfile, and the Ollama server process were all removed afterward; VTE has no runtime or build dependency on it.

It's also worth stating plainly what that number means, because it's the actual point of measuring against Ollama at all: the code doing the dispatching here is **Python**, not C++. Ollama's inference math runs in llama.cpp — hand-tuned C++ GEMV kernels with years of upstream optimization, called through a thin Go server. VTE's Python layer builds the compute graph, resolves kernel arguments, and drives every HIP launch through ctypes — the language usually blamed for making GPU dispatch too slow to compete. Landing at ~89% of llama.cpp's throughput with that much interpreter overhead in the loop is the evidence for this project's core bet: that CPU-side dispatch overhead is an engineering problem you can profile and remove (HIP Graphs, a sampler restricted to top-k, the LM Head captured into the graph — see [Bugs found during development](BUGS.md)), not an inherent tax Python has to pay. The remaining ~11% gap is real GEMV-kernel-efficiency headroom, not "Python being Python."

## Optimizations tried and rejected

Kept disabled rather than removed, so they don't get re-implemented blindly:

- **FFN kernel fusion** (RMSNorm+Gate+Up+SiLU in one launch): measured slower than the unfused version (12.7–13.5 tok/s vs. 18.8 tok/s) — register pressure from holding two accumulators in one loop drops occupancy enough to erase the round-trip savings. Kept behind `VTE_ENABLE_FFN_FUSION` for anyone who wants to re-run the experiment. Full root-cause in [Architecture: Why QKV projection is fused, and why FFN fusion is off](ARCHITECTURE.md#why-qkv-projection-is-fused-and-why-ffn-fusion-is-off).
- **WMMA/Tensor Cores**: rejected early for pure batch=1 GEMV (no benefit at that arithmetic intensity). Worth re-measuring now that batch>1 GEMM is a real, validated code path — not yet done.
