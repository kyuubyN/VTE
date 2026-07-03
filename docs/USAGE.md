[← Back to README](../README.md)

# Getting started: the Python API and desktop UI in detail

Installation and the fastest path to a running desktop app are covered in the [Quick start](../README.md#quick-start) section of the main README. This page covers the plain Python API and desktop-UI specifics that only matter once it's installed.

## Python API

```python
from vte.core.model import VTEModel

model = VTEModel.from_pretrained("qwen2.5:1.5b-q4_k_m", context_length=8192)

for token in model.generate("Summarize the tradeoffs of static batching:", max_tokens=300):
    print(token, end="", flush=True)
```

Switching architectures is just a different `model_name` — everything else about the API is identical (see [Multi-architecture support](GRANITE.md) for what actually changes under the hood):
```python
model = VTEModel.from_pretrained("granite-4.1:3b-q8_0", context_length=8192)

for token in model.generate("Summarize the tradeoffs of static batching:", max_tokens=300):
    print(token, end="", flush=True)
```

Batched decode (same-length prompts only, see [Known limitations](LIMITATIONS.md)):
```python
model = VTEModel.from_pretrained("qwen2.5:1.5b-q4_k_m", max_batch_size=4)

prompts = ["The capital of France is", "2 + 2 equals", "My favorite color is", "Once upon a time"]
for words in model.generate_batch(prompts, max_tokens=200, temperature=0.7):
    print(words)  # one word per sequence, per generation tick
```

## Desktop UI

The UI (chat + live dashboard: tok/s, ms/token, VRAM, model lifecycle, logs; language toggle between Portuguese and English) lives entirely in [vte/ui/app.py](../vte/ui/app.py), a Flet 0.85+ app. Setup and launch are covered in [Quick start](../README.md#quick-start) (`pip install -e .[dev]` then `vte-ui`) — the notes below are specifics that only matter once it's running.

It needs `vte` to be an *importable package*, not just a folder of scripts — running the file directly with a system-wide Python (`python vte/ui/app.py`) fails with `ModuleNotFoundError: No module named 'vte'`, because that Python has never heard of this project. `python -m vte.ui.app` is the fallback if the `vte-ui` console script isn't on `PATH` for some reason — the `-m` runs it *as a module* inside the `vte` package, which is what makes the relative imports resolve; running the file by path does not, even with `.venv` activated.

**Windows-only telemetry dependencies** (`WMI`, `pywin32`, both in `pyproject.toml`) are what let the dashboard read real GPU numbers (dedicated VRAM usage, matching what Task Manager's GPU tab shows) instead of placeholders. Without them the dashboard still works, just with less data. Real GPU temperature needs no extra package — it goes through AMD's ADL (`atiadlxx.dll`, ships with every AMD driver) via [vte/bridge/adl_bridge.py](../vte/bridge/adl_bridge.py) — but only works on an AMD GPU; on anything else (or if ADL fails for any reason) the dashboard shows "N/A" rather than a fabricated number.

The GGUF itself goes in a `Model/` folder — `VTEModel.from_pretrained()` checks it both relative to the current working directory and relative to the repo root, so it's found whether you run from the project folder or invoke `vte-ui` from anywhere else (a fresh shell, a shortcut, another terminal tab). If neither location has it, the error message prints both full paths it checked.
