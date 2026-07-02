"""
Etapa A — Profiling on-device do forward pass (HIP events por categoria).

Roda em modo EAGER (use_hip_graph=False) para que cada kernel passe por
HIPRuntime.launch_kernel, onde o KernelProfiler o envolve com
hipEventRecord(start/stop) e mede o tempo GPU real por categoria.

Warm-up descartado; keepalive desligado para não poluir a medição.
"""
import os
os.environ["VTE_PROFILE"] = "1"

import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from vte.core.model import VTEModel
from vte.bridge.kernel_profiler import PROFILER
from vte.bridge.logger import get_logger

logger = get_logger("EtapaA")

WARMUP = 8
MEASURE = 60


def main():
    logger.info("=== Etapa A: profiling on-device (modo eager) ===")
    model = VTEModel.from_pretrained(
        "qwen2.5:1.5b-q4_k_m", use_hip_graph=False, idle_timeout_seconds=300
    )

    # Desliga o keepalive para a medição (o pulso de 2ms/token e o kernel
    # minúsculo poluiriam tanto o wall-clock quanto a categoria corrente).
    try:
        model._keepalive.pulse = lambda *a, **k: None
    except Exception:
        pass

    prompt = "Once upon a time, in a dark dungeon, a brave knight"
    gen = model.generate(prompt, max_tokens=WARMUP + MEASURE, temperature=0.7)

    n = 0
    measuring = False
    last = None
    for _word in gen:
        n += 1
        if n == WARMUP:
            # Fim do warm-up: zera acumuladores e começa a cronometrar.
            PROFILER.reset_accumulators()
            measuring = True
            last = time.perf_counter()
            continue
        if measuring:
            now = time.perf_counter()
            PROFILER.mark_token((now - last) * 1000.0)
            last = now
        if n >= WARMUP + MEASURE:
            break

    print()
    print(PROFILER.report())


if __name__ == "__main__":
    main()
