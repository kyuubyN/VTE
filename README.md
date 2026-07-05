<p align="center">
  <img src="assets/earth_pixelated.svg" width="96" height="96" alt="VTE logo">
</p>

<h1 align="center">VTE — Vector Tensor Engine</h1>

<p align="center">
  A from-scratch LLM inference engine for AMD GPUs on Windows.<br>
  No llama.cpp. No PyTorch. No ONNX Runtime. Just HIP, GGUF, and a lot of measurement.
</p>

<p align="center">
  <img alt="platform" src="https://img.shields.io/badge/platform-Windows-0078D6">
  <img alt="gpu" src="https://img.shields.io/badge/GPU-AMD%20RDNA2%2F3-ED1C24">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-3776AB">
  <img alt="license" src="https://img.shields.io/badge/license-Apache%202.0-blue">
  <img alt="vte-vs-ollama" src="https://img.shields.io/badge/vs.%20llama.cpp-88.7%25%20throughput-00A36C">
</p>

---

VTE parses the GGUF file itself, generates the HIP C++ kernels itself, compiles them with `hipcc` at runtime, and drives `amdhip64.dll` through a hand-written ctypes bridge. It currently runs two architectures — **Qwen2.5-1.5B-Instruct** (Q4_K_M/Q6_K) and **IBM Granite 4.1 3B** (Q8_0) — selectable at runtime, and the whole project has been developed and measured on a single consumer card: an RX 7600 (RDNA3, gfx1102, 8GB VRAM).

The reason to build this from scratch was to have full control over every byte moved between VRAM and the ALUs on a GPU that has neither the memory bandwidth nor the CU count of a datacenter part — and to make every optimization decision based on an actual measurement on this specific hardware, not on what works on an MI300X or an RTX 4090. That discipline ("measure, don't guess") shows up throughout the codebase and is documented in [Bugs found during development](docs/BUGS.md).

As of this writing: single-sequence decode holds a **stable ~100 tok/s on Qwen2.5-1.5B**, and batched decode peaks at **~200 tok/s aggregate** at batch size 4 — both climbed from a ~41 tok/s baseline through profiling, not GEMV rewrites. Granite 4.1 3B, added as a test of whether the architecture scales past a 1.5B toy model, runs correctly in ~4.8GB VRAM and reaches the same throughput ratio vs. llama.cpp that Qwen does. Full numbers below.

## Benchmark: VTE vs. Ollama (llama.cpp)

Same GGUF files on disk for both engines, same prompt, `temperature=0`, decode-only timing (full methodology in [Performance](docs/PERFORMANCE.md)):

| Model | VTE | Ollama (llama.cpp) | VTE / Ollama |
|---|---|---|---|
| Qwen2.5 1.5B (Q4_K_M) | 101.68 tok/s (9.84 ms/tok) | 114.67 tok/s (8.72 ms/tok) | **88.7%** |
| Granite 4.1 3B (Q8_0) | 45.66 tok/s (21.90 ms/tok) | 51.46 tok/s (19.43 ms/tok) | **88.7%** |

Notably consistent — an identical ratio across two architectures with almost nothing in common (different RoPE convention, different quantization format, ~2x the parameters). The code doing the dispatching here is **Python**, not C++, driving every HIP launch through ctypes — landing at ~89% of a mature, years-tuned C++ engine's throughput is the evidence for this project's core bet: that dispatch overhead is a solvable engineering problem, not a language tax. See [Performance](docs/PERFORMANCE.md#benchmark-vte-vs-ollama-llamacpp) for the full write-up.

## Quick start

The fastest way to talk to a model is the desktop UI (chat + live GPU telemetry).

<table>
<tr><td>

**Requirements**
- Windows 10/11, 64-bit
- AMD RDNA2/RDNA3 GPU (RX 6000/7000 series)
- HIP SDK (ROCm 6.4) + MSVC Build Tools
- Python 3.10+, ~8GB VRAM
- A `.gguf` model in `Model/` — Qwen2.5-1.5B or Granite 4.1 3B

</td><td>

**Install and run**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
vte-ui
```

</td></tr>
</table>

`pip install -e .[dev]` registers `vte` as an editable package (`import vte` and `vte-ui` work from anywhere afterward). Pick the model from the dropdown in the app — switching between Qwen and Granite at runtime, mid-session, is supported (see [Multi-architecture support](docs/GRANITE.md)).

If `vte-ui` isn't on `PATH`, `python -m vte.ui.app` is the equivalent. To install VTE as a library without cloning: `pip install git+https://github.com/kyuubyN/VTE.git`. Full requirements list, Python API examples, and desktop-UI details: [Getting started](docs/USAGE.md).

## Documentation

The full write-up is split by topic — each page is self-contained and links back here.

| Page | What's in it |
|---|---|
| [**Performance**](docs/PERFORMANCE.md) | Stage-by-stage optimization history (18.8 → 100 tok/s), batched decode, the full VTE-vs-Ollama benchmark, and what was tried and rejected |
| [**Architecture**](docs/ARCHITECTURE.md) | How `vte/bridge`, `vte/compiler`, `vte/core`, and `vte/ui` fit together; why QKV fusion is on but FFN fusion is off; notes for AMD/ROCm reviewers |
| [**Multi-architecture support**](docs/GRANITE.md) | Adding Granite 4.1 3B: the RoPE convention bug, the `residual_scale` scoping bug, and the Flet UI's model-switch race conditions |
| [**Qwen 3.5 (hybrid Gated DeltaNet)**](docs/QWEN35.md) | Adding a hybrid recurrent-attention architecture: the FP16-read-as-float32 bug that caused "oi" to degenerate into garbage, the missing QK-Norm/gate architecture piece, and the streaming/thinking-mode bugs found via real UI testing |
| [**Bugs found during development**](docs/BUGS.md) | The full "symptom → root cause → fix → measurement" history — silent data corruption, a real Windows TDR crash, and everything in between |
| [**Known limitations**](docs/LIMITATIONS.md) | What's genuinely unfinished or unresolved right now |
| [**Getting started**](docs/USAGE.md) | Full Python API examples and desktop-UI setup details |
| [**Security policy**](SECURITY.md) | Threat model and defense mechanisms (untrusted GGUF input, VRAM sandboxing, watchdogs) |

---

<p align="center">VTE (Vector Tensor Engine) — 2026 · Licensed under Apache 2.0</p>
