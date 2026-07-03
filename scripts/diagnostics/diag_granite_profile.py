"""
Perfil por categoria (RMSNorm, FusedQKV, FlashAttention, FFN, LMHead, etc.)
do forward pass do Granite -- mesma metodologia usada para otimizar o Qwen
(Split-KV, QKV Two-Pass Split-K) nesta sessao. Roda com VTE_PROFILE=1 para
medir onde o tempo de GPU realmente vai, em vez de otimizar as cegas.
"""
import os
os.environ["VTE_PROFILE"] = "1"

from vte.core.model import VTEModel
from vte.bridge.kernel_profiler import PROFILER

m = VTEModel.from_pretrained("granite-4.1:3b-q8_0", context_length=512, use_hip_graph=False)
prompt = m.tokenizer.apply_chat_template(
    "Escreva um texto longo e detalhado sobre a historia da exploracao espacial."
)

WARMUP = 10
MEASURE = 60

gen = m.generate(prompt, max_tokens=WARMUP + MEASURE, temperature=0.7, top_p=0.9)

import time
count = 0
for w in gen:
    count += 1
    if count == WARMUP:
        PROFILER.reset_accumulators()
    if count >= WARMUP + MEASURE:
        break

total = sum(PROFILER.gpu_ms.values())
print(f"=== {MEASURE} tokens medidos (eager/FallbackExecutor) ===")
print(f"{'Categoria':<20}{'GPU ms total':>14}{'ms/tok':>10}{'%':>8}{'launches/tok':>14}")
for cat in sorted(PROFILER.gpu_ms, key=lambda c: -PROFILER.gpu_ms[c]):
    ms = PROFILER.gpu_ms[cat]
    pct = 100.0 * ms / total if total else 0.0
    print(f"{cat:<20}{ms:>14.2f}{ms/MEASURE:>10.3f}{pct:>7.1f}%{PROFILER.counts[cat]/MEASURE:>14.1f}")
print(f"{'TOTAL':<20}{total:>14.2f}{total/MEASURE:>10.3f}")
