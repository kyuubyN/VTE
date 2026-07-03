# VTE — Vector Tensor Engine

VTE is an inference engine for running LLMs on AMD GPUs on Windows, built directly on top of the HIP runtime instead of any existing inference framework. There's no llama.cpp, no ONNX Runtime, no vLLM, no PyTorch underneath it — VTE parses the GGUF file itself, generates the HIP C++ kernels itself, compiles them with `hipcc` at runtime, and drives `amdhip64.dll` through a hand-written ctypes bridge. The only thing it currently runs is Qwen2.5-1.5B-Instruct in Q4_K_M quantization, and the whole project has been developed and measured on a single consumer card: an RX 7600 (RDNA3, gfx1102, 8GB VRAM).

The reason to build this from scratch rather than wrapping an existing runtime was to have full control over every byte moved between VRAM and the ALUs on a GPU that has neither the memory bandwidth nor the CU count of a datacenter part — and to make every optimization decision based on an actual measurement on this specific hardware, not on what works on an MI300X or an RTX 4090. That discipline ("measure, don't guess") shows up throughout the codebase and is documented below alongside the bugs it caught.

As of this writing, the engine does single-sequence decode at up to **~71.7 tok/s** (steady-state, 149-token sample) and batched decode (multiple sequences in lockstep) peaking at ~200 tok/s aggregate throughput at batch size 4, both validated with bit-exact numeric diffs against a NumPy reference implementation. The single-sequence number nearly doubled from an earlier ~41 tok/s baseline during a single investigation session — not by touching a single HIP kernel, but by profiling the actual per-token time budget and finding that the CPU-side sampler and an eager (non-graph-captured) LM Head launch were together costing more than the GPU's own 28-layer forward pass. See "Bugs found during development" below for the full story, including a real Windows TDR crash that turned out to have nothing to do with the GPU.

Batched decode was independently re-measured and reconfirmed after a full repository reorganization (200.7 tok/s aggregate / 50.2 tok/s per sequence at batch_size=4, over 100 decode ticks) — these numbers reflect the current state of the code, not a one-off historical run. That said, the ~200 tok/s figure is a GPU-only measurement (HIP Graph replay in isolation); the same eager-LM-Head/sampler overhead found in single-sequence decode was also present in batched decode, and applying the same fix there is documented in "Bugs found during development" below, alongside a correction to how large a win it actually was.

## Performance

| Stage | Throughput (batch=1) | What changed |
|---|---|---|
| Naive per-thread GEMV | ~18.8 tok/s | Baseline: one thread per output element, no coalescing |
| Coalesced GEMV | ~31.5 tok/s | Threads in a block cooperate on one output neuron, coalesced weight reads |
| In-kernel Q4_K/Q6_K dequantization | ~37.9 tok/s | FFN weights dequantized inline during the GEMV instead of a separate dequant pass ("No-Sync Direct Unpack") |
| QKV Two-Pass Split-K fusion | ~41.0 tok/s | Fills all 32 CUs instead of leaving most idle — see "QKV fusion" below |
| Vectorized sampler | ~62–70 tok/s | Top-k/top-p/softmax/argsort restricted to the ~50 surviving candidates instead of the full 151936-token vocabulary — see bug entry below |
| LM Head captured into the HIP Graph | **~71.7 tok/s** | Eliminated the last eager (non-graph) kernel launch in the decode loop — see bug entry below |

None of the last two rows touched a single HIP kernel — both came from profiling the actual per-token wall-clock budget (not just "the GPU part") and finding that CPU-side and dispatch overhead, not GEMV bandwidth, was the larger cost. The GPU-only decode graph itself measures ~130 tok/s in isolation (via HIP Graph replay timing) — the fact that end-to-end throughput is still well below that is the next thing being investigated (see "Notes for anyone reviewing this from an AMD/ROCm perspective" below).

Batched decode (`generate_batch`, same-length prompts, lockstep) — GPU-only, HIP Graph replay in isolation (no sampler, no logits readback):

| Batch size | Aggregate tok/s | Efficiency vs. ideal linear scaling |
|---|---|---|
| 1 | 110.7 | — |
| 2 | 165.3 | 75% |
| 4 | 204.2 (peak) | 46% |
| 8 | 193.5 (regresses) | 22% |

The batch=8 regression is attributed to Infinity Cache (32MB) thrashing: FFN weights are ~27.5MB per layer (~88% of per-layer traffic), and at batch=8 the working set for weight reuse across the batch no longer fits comfortably alongside everything else competing for the cache. This hasn't been fixed yet — see the open items at the end of this document.

**Real end-to-end `generate_batch()` throughput at batch_size=4** (sampler + logits readback included, the number that actually matters for a caller): **127.6 tok/s aggregate** (31.9 tok/s/sequence), up from 118.9 tok/s after capturing the LM Head into the batched HIP Graph (same technique as single-sequence decode, adapted to `[batch_size, vocab_size]` geometry) — a real but modest 7.4% gain, smaller than a naive Amdahl's-law projection suggested. See "Bugs found during development" for why: most of the eagerly-measured LM Head cost turned out to be genuine GPU work (the model's single largest GEMV, batched 4-wide), not removable dispatch overhead.

Two things that were tried and measured as *not* worth it, kept disabled rather than removed, so they don't get re-implemented blindly:
- **FFN kernel fusion** (RMSNorm+Gate+Up+SiLU in one launch): measured slower than the unfused version (12.7–13.5 tok/s vs. 18.8 tok/s) — register pressure from holding two accumulators in one loop drops occupancy enough to erase the round-trip savings. Kept behind `VTE_ENABLE_FFN_FUSION` for anyone who wants to re-run the experiment.
- **WMMA/Tensor Cores**: rejected early for pure batch=1 GEMV (no benefit at that arithmetic intensity). Worth re-measuring now that batch>1 GEMM is a real, validated code path — not yet done.

## Requirements

- **OS:** Windows 10/11, 64-bit (the engine talks to WDDM directly for TDR-related timing behavior; Linux/ROCm is not tested)
- **GPU:** AMD RDNA2/RDNA3 (RX 6000/7000 series). Built and measured on an RX 7600 (gfx1102, 8GB VRAM)
- **HIP SDK:** ROCm 6.4 (the codebase probes `6.4` down to `5.7` under `C:\Program Files\AMD\ROCm\`, but 6.4 is what's actually been tested)
- **Native compiler:** Microsoft C/C++ Build Tools (`cl.exe`) — `hipcc` on Windows doesn't locate the MSVC/Windows SDK headers on its own; VTE's codegen layer injects the right `PATH`/`INCLUDE` entries manually before invoking it
- **Python:** 3.10+
- **VRAM:** ~8GB observed at `batch_size=1`; batched decode needs extra headroom for the batch-strided KV cache
- **Model file:** Qwen2.5-1.5B-Instruct, GGUF, `Q4_K_M` quantization

Install:
```bash
pip install git+https://github.com/kyuubyN/VTE.git
```

## How it fits together

Three packages, roughly bottom-up:

**`vte/bridge`** — the only layer that touches the AMD driver directly.
- `hip_runtime.py`: a ctypes wrapper around `amdhip64.dll`. Every allocation is tracked, every pointer is bounds-checked before a memcpy, grid/block/shared-memory dimensions are validated against the real device limits before every launch, and device→host copies are blocked unless the destination tag is explicitly allowlisted (an intentional data-isolation barrier). It also owns two mechanisms that exist specifically because this is a *consumer* GPU shared with the rest of the user's desktop, not a dedicated datacenter accelerator: a `KernelWatchdog` that puts the runtime into a global panic mode if any single launch or `synchronize()` takes too long (protects against a truly hung kernel without doing a risky mid-flight device reset), and a duty-cycle limiter that inserts small preventive pauses before dispatch to keep sustained GPU utilization near 95% instead of pegging it at 100% indefinitely.
- `memory.py`: a slab allocator — one big `hipMalloc` up front, then best-fit sub-allocation with free-list merging inside it, so the engine never pays repeated `hipMalloc`/`hipFree` overhead or fragmentation during a generation loop.
- `gpu_utilization_guard.py`, `kernel_profiler.py`, `watchdog.py`, `dll_discovery.py`, `logger.py`: supporting instrumentation — an independent (Windows performance-counter based) utilization sample used only for diagnostics, on-device GPU-time profiling via HIP events (isolating kernel time from CPU dispatch overhead), the panic-mode watchdog itself, and DLL/toolchain discovery.

**`vte/compiler`** — turns a GGUF file into HIP kernels and a memory layout, ahead of any generation.
- `sanitizer.py` + `gguf_parser.py`: two layers of validation against untrusted `.gguf` input before any tensor is parsed for real (file size/hash/magic/version, tensor bounds).
- `dequantizer.py`: byte-exact Q4_K/Q6_K → FP32/FP16 dequantization matching llama.cpp's reference bit layout — this is where two of the nastier historical bugs lived (see below).
- `weight_loader.py`: mmaps the GGUF and uploads weights to VRAM, either dequantized to FP16 or left raw (Q4_K/Q6_K) for tensors routed to in-kernel dequantization kernels, in host→device chunks small enough to avoid tripping Windows' TDR watchdog on a single giant transfer.
- `qwen_mapper.py`: computes the full VRAM memory plan (weights, KV cache, activation arena, RoPE cache) and maps every tensor to an address inside the slab. This is the single most bug-dense file in the project's history — KV cache layout mistakes here silently corrupted attention outputs, not crashed.
- `ir.py`, `qwen_compute.py`, `fusion_analyzer.py`, `fusion_applier.py`: build a small IR graph of the model's operations (RMSNorm → MatMul → RoPE → Attention → SwiGLU → …) and look for chains that can be legally fused into a single kernel, given RDNA3 VGPR/LDS limits.
- `codegen.py`: renders `.hip.template` files with sanitized parameter substitution (this is the one place where string interpolation into compiled C++ happens, and it's treated as such), invokes `hipcc`, caches binaries by content hash, and rejects any kernel whose VGPR usage risks register spilling.
- `tokenizer.py`: a from-scratch byte-level BPE tokenizer that reconstructs Qwen2.5's vocabulary and merge rules straight from the GGUF file, replicating the GPT-2/tiktoken algorithm without depending on an external tokenizer library.

**`vte/core`** — orchestrates a generation request end to end.
- `model.py` (`VTEModel`): the public entry point (`from_pretrained` / `generate` / `generate_batch`). Loads real hyperparameters from GGUF metadata (several early defaults were simply wrong for this model — see bugs below), builds the compute graph, applies kernel fusion, and picks an executor.
- Two executor families, each with a batched variant built by composition rather than inheritance:
  - `fallback_executor.py` / `batched_fallback_executor.py`: eager, node-by-node dispatch through the IR graph. Easier to debug, used as the safety net if graph capture fails.
  - `hip_graph_executor.py` / `batched_hip_graph_executor.py`: captures the entire per-token decode path into a HIP Graph once and replays it, eliminating almost all Python-side dispatch overhead. Static addresses required for graph capture are why the engine pre-allocates persistent activation buffers for every intermediate tensor up front (see "known limitation" below). The LM Head's final GEMV is captured into this same graph (`_capture_lm_head` / `_capture_lm_head_batch`) rather than launched eagerly after each replay, in both the single-sequence and batched executors — `model.py` resolves the LM Head's weight pointer (respecting tied embeddings), logits buffer (`[vocab_size]` or `[batch_size, vocab_size]`), and compiled kernel *before* the respective executor is constructed, specifically to have those fixed addresses ready at capture time. The graph only ever *writes* logits into VRAM — the device→host copy and CPU-side sampling always happen after `graph_launch()` returns, never inside the captured stream.
- `fused_qkv_dispatch.py`: the QKV+RoPE and FFN Gate+Up+SiLU fusion logic shared by both executor families (see "QKV fusion" below for why this exists and what it replaced).
- `kernel_arg_builder.py`: builds the exact `void**` argument arrays HIP kernel launches require, per operation type — this is where a couple of the "silent data corruption" bugs below were actually rooted (an argument count or shape mismatch here doesn't crash, it just feeds a kernel plausible-looking garbage).
- `generator.py`, `sampler.py`, `lm_head.py`: the autoregressive loop (tokenize → prefill → decode-until-stop), CPU-side sampling (repetition penalty, temperature, top-k/top-p, stable softmax — restricted to the top-k candidate subset, not the full vocabulary), and the final hidden-state → logits projection (normally captured into the HIP Graph; `LMHead` keeps an eager fallback path used only when HIP Graph capture fails or is disabled).
- `lifecycle.py`, `motor.py`, `ipc.py`, `gpu_monitor.py`: idle-timeout auto-unload of the model from VRAM, the backend process that the desktop UI talks to over a local pipe, and GPU telemetry (VRAM/temperature/utilization) for the dashboard.

**`vte/ui`** — a Flet desktop app (chat panel + live GPU telemetry dashboard) talking to the `motor.py` backend process over a local pipe.

## Bugs found during development, and what they looked like

Most of the interesting bugs in this project didn't crash — they produced plausible-looking, silently wrong output, which is the worst failure mode for a numerical system like this one. Every one of these was eventually caught by comparing intermediate activations against a NumPy reference implementation layer by layer, not by staring at the code. That's the main reason the `diag_*.py` scripts at the repo root exist as a set of throwaway-but-preserved investigation scripts rather than being deleted after each bug was fixed.

- **Wrong hyperparameters from hardcoded defaults.** Several defaults scattered through early code were simply wrong for Qwen2.5-1.5B: `head_count=16` instead of the real 12, `rope.freq_base=1e4` instead of the model's actual `1e6`, `eps=1e-5` instead of `1e-6`. Wrong head count breaks GQA head-group math and launches attention blocks out of bounds; wrong RoPE base silently rotates embeddings by the wrong angle at every position beyond the first. Fixed by reading every hyperparameter from the GGUF's own metadata instead of a hardcoded table.
- **KV cache V overlapping K.** An early KV cache layout put the V cache in the *second half* of K's allocated space and advanced the per-layer offset using only half the actual size needed. Every layer after the first silently corrupted the layer before it. Fixed by reserving `2x` the per-layer size (K and V distinct) and, once batching was added, multiplying by `batch_size` since each sequence needs its own K/V space from the first token onward.
- **`context_length` metadata mismatch.** The KV cache offset stride must be computed from the *runtime* `context_length` parameter, not the model's native GGUF context metadata (32768 for Qwen2.5, ~16x the old default of 2048). Using the wrong one made the per-layer offset advance faster than the allocated pool, overrunning into the activation arena and subsequent buffers.
- **Input tokens never reaching VRAM.** `_write_input_ids` was, at one point, not actually called before the embedding lookup — the model was processing whatever bytes happened to already be in that buffer, completely ignoring the prompt, and doing so without any error at all.
- **SwiGLU zeroing the FFN.** The SwiGLU kernel takes one `total_elements` argument (`batch * seq_len * intermediate_size`), but the argument builder was passing shape components as three separate arguments instead — the kernel read `total_elements=batch=1` and processed exactly one of the 8960 FFN elements, discarding the rest of the signal on every forward pass.
- **LM head logits reinterpreted at the wrong width.** `matmul_kernel` writes its output in FP16, but the logits buffer was allocated and read back as FP32. The host was reinterpreting pairs of adjacent FP16 values as single FP32 values, producing logits with magnitude ~1e7 instead of the real ~±40 range — with a perfectly correct hidden state feeding into it.
- **Kernel cache key collisions in the HIP Graph path.** Two MATMULs of the same shape but routed to different kernel templates (e.g. a Q4_K `down_proj` via `gemv_q4k` vs. an FP16 `attn_output` via `gemv_coalesced`) hashed to the same cache key when the key was `(op_type, shape)` instead of including the template name. One node would silently inherit the other's compiled kernel and read a quantized weight as if it were raw FP16 — a bug that only manifested in the graph-capture path, since the eager executor happened to key its own cache by name.
- **Embedding lookup silently frozen across graph replays.** The compute graph's `INPUT` node is a marker, always skipped by the dispatcher — `FallbackExecutor` had a manual embedding-lookup step outside the graph loop, but `HIPGraphExecutor` never called an equivalent, so every replay reused whichever embedding happened to be captured once, regardless of the actual token being generated.
- **Q4_K scale/min index bug.** In the sub-block scale/min extraction (`_q4k_scale_min`, replicating llama.cpp's `get_scale_min_k4`), the branch for `j>=4` needs the high 2 bits of `sc` from `scales[j-4]>>6` but the high 2 bits of `m` from `scales[j]>>6` — not `scales[j-4]` for both. Using `j-4` for both happened to work when those particular bits were equal, and silently corrupted part of the dequantized weights when they weren't.
- **Q4_K nibble ordering.** An earlier dequantization pass mixed the low/high nibbles of each byte in 16-element halves instead of the correct 32-element grouping per sub-block, producing weights of plausible magnitude but scrambled values — a model that "ran" and produced fluent-looking garbage.
- **A hardcoded 15ms-per-token floor that became the bottleneck.** A fixed keep-alive delay between decode steps was originally harmless, but after HIP Graph capture and kernel fusion brought the real per-token GPU time well under that floor, the fixed delay became an artificial throughput ceiling (~66 tok/s) that masked the actual gains from every subsequent optimization. Replaced with a much smaller (2ms) keep-alive pulse, just enough to smooth clock-state transitions without becoming the new bottleneck itself.
- **A kernel-argument-count mismatch that looked like a hardware fault.** Two GEMV kernel templates (`gemv_coalesced`, `gemv_q4k`/`gemv_q6k`) take 9 parameters, the last being a `residual_ptr` added later for the epilogue-fusion optimization. A couple of diagnostic scripts were never updated after that signature change and called the kernel with only 8 arguments. Because the runtime's `expected_args` check only validates a caller's own argument-list length against a caller-supplied number — it has no way to know the real compiled kernel's signature — the mismatch went undetected at the Python level, and the kernel read uninitialized memory for the missing parameter: sometimes producing silently wrong output, sometimes dereferencing a garbage pointer and hard-faulting the GPU (a real `VIDEO_ENGINE_TIMEOUT_DETECTED` Windows TDR event). It took ruling out the GPU, the driver, a full OS reboot, and a full ROCm reinstall — all innocent — before isolating it to this one stale call site. The lesson: when a kernel template's parameter list changes, grep for every call site of that kernel; `expected_args` is a self-consistency check, not proof a call site matches the real kernel.
- **Applying the same LM-Head-in-graph fix to batched decode revealed the limits of a naive Amdahl's-law projection.** The same eager-LM-Head pattern existed in `generate_batch()`'s `compute_logits_batch()` call. Measured in isolation, it cost ~7.5ms/tick — the obvious hypothesis was "that's dispatch overhead, capturing it into the batched HIP Graph should recover most of it," projecting end-to-end throughput back up near the ~200 tok/s GPU-only ceiling. After implementing the capture (validated bit-exact against the eager path across all 4 sequences, same as the single-sequence case), the batched decode graph + LM Head fused together measured 24.4ms/tick — barely faster than the 18.5ms (decode) + 7.5ms (eager LM Head) = 26.0ms sum, recovering only ~1.7ms. The correction: most of that 7.5ms was never removable dispatch overhead in the first place — it's the model's single largest GEMV (vocab=151936 × hidden=1536, batched 4-wide, ~933M multiply-adds), and that's genuine GPU work whether it's launched eagerly or captured in a graph. End-to-end batched throughput went from 118.9 to 127.6 tok/s aggregate (+7.4%) — a real, validated win, just a much smaller one than the initial back-of-envelope math suggested. The lesson: a measured cost being "outside the graph" doesn't mean all of it is dispatch overhead — check whether the underlying kernel is cheap (mostly overhead, like the small attention-weight GEMVs elsewhere in this document) or expensive (mostly real work, like this one) before projecting how much a graph capture will actually recover.
- **A measurement protocol that quietly outweighed what it was measuring.** While chasing the 41→100 tok/s gap, an isolated GEMV microbenchmark timed each kernel launch individually (create/record/launch/record/read a HIP event pair, synchronizing after every call — the safe pattern learned from the TDR above). It showed *zero* speedup from routing attention weights through the in-kernel Q4_K/Q6_K dequant kernels, even though the same technique clearly helped the FFN. The reason: at attention's small shapes (K≤1536), the real GPU work is a handful of microseconds, but the per-call host/driver round-trip protocol cost a fixed ~180–200µs regardless of what the kernel actually did — confirmed by timing a trivially small kernel and getting the same ~186µs. The fix wasn't the kernel, it was the ruler: capturing N repetitions of the same kernel into one HIP Graph and timing the whole replay (one round-trip amortized over N real executions) showed the true per-kernel cost was ~6–12µs — and that quantizing attention weights is neutral-to-slightly-negative there, correctly ruling out that whole optimization path with real numbers instead of a noise floor.
- **The actual bottleneck wasn't the GPU at all.** The same graph-replay timing technique, applied to the real production decode graph, showed ~7.6ms/tok of genuine GPU work (a ~130 tok/s ceiling) against a ~13.7–22.8ms/tok wall-clock — meaning something *outside* the captured graph was costing more time than the 28-layer forward pass itself. Decomposing it (timing each piece of the per-token Python loop in isolation) found two culprits: the CPU-side sampler doing `argsort`/`softmax`/`cumsum` over the full 151936-token vocabulary when only the top-k survivors (~50) mattered after filtering (9.4ms, the single largest component measured — larger than the GPU work), and the LM Head's final GEMV running as an eager kernel launch outside the HIP Graph (~3.2ms, paying dispatch overhead the graph exists to eliminate). Fixing the sampler to restrict every downstream operation to the top-k subset before sorting/softmax/sampling cut it to ~0.9ms (bit-exact on the greedy path, statistically identical distribution otherwise — validated before trusting any speed number). Capturing the LM Head into the same decode graph (resolving its tied-embedding weight pointer, logits buffer, and compiled kernel *before* the graph executor is constructed, to have fixed addresses ready at capture time) removed the remaining eager launch. Together: ~41 → ~71.7 tok/s, without touching a single GEMV kernel.

## Why QKV projection is fused, and why FFN fusion is off

These two decisions look inconsistent at a glance (fuse one, not the other) but both came from measurement, not intuition:

The original QKV design launched one kernel block per attention head (12 heads for Q, 2 for K/V) — on a 32-CU GPU, that leaves most compute units idle (occupancy-bound, not ALU-bound). The current design (Two-Pass Split-K) does a single shared RMSNorm, then splits each projection's inner dimension across 32 blocks regardless of head count, filling every CU. Measured: 37.9 → 41.0 tok/s (+8%), validated numerically and end-to-end with real text.

FFN fusion (RMSNorm+Gate+Up+SiLU as one launch) was built the same way and measured *slower*: 18.8 tok/s unfused vs. 12.7–13.5 tok/s fused. The FFN's grid only has ~35 blocks (`intermediate_size / 256`), and holding two running accumulators (gate and up) in the same loop increases per-thread register pressure enough to drop occupancy below the point where fewer VRAM round-trips pay for themselves. It's kept in the codebase, disabled by default, behind `VTE_ENABLE_FFN_FUSION`, specifically so this negative result doesn't get silently rediscovered and re-implemented by someone who didn't see the measurement.

## Security posture

The only input to this project that should be treated as genuinely untrusted is a `.gguf` file, since these are commonly downloaded from third-party sources. It goes through two validation layers (`GGUFSanitizer` then `GGUFParser`) before any tensor data is used: file size/hash/magic/version checks, tensor count/KV count caps, and per-tensor offset/size bounds checking against the actual file. Two things that look like gaps but are intentional, documented design decisions: the GPU utilization guard is observation-only by design (it never kills in-flight work), and the UI↔motor IPC pipe has no authentication (local-only threat model, not designed to cross a network boundary).

## Notes for anyone reviewing this from an AMD/ROCm perspective

A few things in this codebase exist specifically because of gaps or friction encountered targeting a consumer RDNA3 part through HIP on Windows, rather than a datacenter part through the usual ROCm software stack. Flagging them here in case any have a better-known solution, or are useful data points either way:

- **`hipcc` doesn't find MSVC/Windows SDK headers on its own on Windows** — `codegen.py`'s `_setup_hip_env` manually injects `PATH`/`INCLUDE` entries before invoking it, or compilation fails with `cmath not found`. This felt like something that should be handled by the SDK installer or `hipcc` itself.
- **A single large synchronous `hipMemcpy` (Host→Device) is a practical TDR trigger on Windows/WDDM.** Uploading `token_embd.weight` (~445MB) in one call was enough to risk the driver considering the GPU hung. `weight_loader.py` chunks all host→device uploads into ≤16MB pieces specifically to give the driver windows between calls. It's not clear whether this is a WDDM constraint that a well-behaved HIP application on Windows is just expected to work around, or something the driver could handle more gracefully.
- **No official signal for "this consumer GPU is a shared desktop resource, keep it under X% utilization sustained."** `HIPRuntime`'s duty-cycle limiter (`_throttle_before_dispatch` / `_enforce_duty_cycle_limit`) is a from-scratch mechanism that measures real busy-time in a sliding window and inserts pauses to keep sustained utilization near 95%, specifically so the engine doesn't make the rest of the user's desktop unresponsive during long generations. There's no ROCm/driver-level equivalent being used here — this is pure userspace timing.
- **VGPR/register-spill risk is checked by parsing `hipcc`'s own compiler output** (`codegen.py::_parse_vgpr_usage`), not queried through any structured API — kernels whose reported VGPR usage exceeds 128 are rejected before ever being launched. A structured way to query this (occupancy calculator equivalent for arbitrary generated kernels) would remove some fragility here.
- **The FFN fusion regression (above) is architecture-specific to RDNA3's 32 CUs at this problem size** (`intermediate_size=8960` → ~35 blocks). It's plausible this fusion would actually win on a part with a very different CU count/register file ratio; the measurement here should be read as "wrong for this specific GPU," not as a general claim about the fusion technique.
- **HIP Graphs eliminate CPU-side dispatch overhead but not GPU-side kernel-boundary cost** (weight-streaming interruption + restart latency + intermediate HBM round-trips between graph nodes). There's an open, not-yet-measured question about how large that boundary cost actually is on a 32-CU RDNA3 part, and whether fusing a few more adjacent stages would be worth it here specifically.
- **A single-sequence decode step still uses only a fraction of the RX 7600's theoretical memory bandwidth — but the gap is now isolated to the GEMV kernels themselves, not overhead around them.** Rough math: at ~288 GB/s theoretical GDDR6 bandwidth and ~1.14GB of weights read per token, the physical floor is ~4ms/token (~250 tok/s ceiling). Measured via HIP Graph replay (timing N repeated replays of the real decode graph, isolating GPU time from any host-side cost), the 28-layer decode graph itself runs at ~7.6ms/tok — a ~130 tok/s ceiling, about 52% of the theoretical floor. End-to-end throughput (~71.7 tok/s) is now close enough to that GPU-only ceiling that the remaining gap is mostly the graph's own kernels, not dispatch or CPU-side overhead (both of those were found and fixed — see "Bugs found during development"). Closing the ~130 vs ~250 gap further would mean comparing kernel-level design choices (block size, load pattern, LDS usage) against other HIP/ROCm GEMV implementations on the same hardware — not yet done; any insight into what a well-tuned RDNA3 GEMV should be capable of in practice (vs. datasheet bandwidth) would be useful context.

## Known limitations

- `generate_batch` requires every prompt in a batch to have identical token length (lockstep only); padding + attention masking for mixed-length batches is unimplemented.
- Batch sizes above 4 currently regress throughput (see the performance table above) — not yet root-caused beyond the Infinity Cache hypothesis.
- Persistent activation buffers used by the HIP Graph executor are sized for `seq_len=1` (the dominant decode case); a multi-token prefill through that executor could in principle overrun those buffers. The current code avoids this by processing prefill token-by-token even in HIP Graph mode, reusing the single decode graph — but the buffers themselves aren't yet partitioned by shape class to make this impossible by construction rather than by calling convention.
- FFN kernel fusion and WMMA/Tensor Cores remain measured-and-rejected (former) or unevaluated in the current batch>1 regime (latter) — see the performance section.
- Single-sequence decode throughput (~71.7 tok/s) is still below the GPU's theoretical memory-bandwidth ceiling (~250 tok/s) and the decode graph's own measured GPU-only ceiling (~130 tok/s) — the CPU/dispatch overhead that used to dominate this gap has been found and fixed (sampler, LM Head), so what remains is genuinely about GEMV kernel efficiency, not overhead around it. Not yet closed.
- `LMHead`'s HIP-Graph capture path assumes the decode graph always runs at `seq_len=1` (true for the current lockstep decode/prefill scheme) — a future multi-token-per-step scheme would need the LM Head capture logic revisited alongside it.

## Getting started

```python
from vte.core.model import VTEModel

model = VTEModel.from_pretrained("qwen2.5:1.5b-q4_k_m", context_length=8192)

for token in model.generate("Summarize the tradeoffs of static batching:", max_tokens=300):
    print(token, end="", flush=True)
```

Batched decode (same-length prompts only, see "Known limitations"):
```python
model = VTEModel.from_pretrained("qwen2.5:1.5b-q4_k_m", max_batch_size=4)

prompts = ["The capital of France is", "2 + 2 equals", "My favorite color is", "Once upon a time"]
for words in model.generate_batch(prompts, max_tokens=200, temperature=0.7):
    print(words)  # one word per sequence, per generation tick
```

Desktop UI (requires a `Model/` directory containing the GGUF):
```bash
vte-ui
```

---

VTE (Vector Tensor Engine) — 2026
Licensed under Apache 2.0
