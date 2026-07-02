"""Etapa B — gate/up vetorizado: velocidade + coerência em modo HIP Graph."""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from vte.core.model import VTEModel
from vte.bridge.logger import get_logger
logger = get_logger("EtapaB")


def main():
    model = VTEModel.from_pretrained("qwen2.5:1.5b-q4_k_m", use_hip_graph=True, idle_timeout_seconds=120)
    prompt = "Once upon a time, in a dark dungeon, a brave knight"
    logger.info(f"Prompt: '{prompt}'")
    start = time.perf_counter()
    gen = model.generate(prompt, max_tokens=200, temperature=0.7)
    toks, full, cap = [], [], None
    for w in gen:
        toks.append(w); full.append(w)
        n = len(toks)
        if n == 1: cap = time.perf_counter() - start
        if n % 50 == 0:
            print(f"--- {n} tok, {n/(time.perf_counter()-start):.1f} tok/s ---", flush=True)
    total = time.perf_counter() - start
    print(f"=== DONE {len(toks)} tok em {total:.1f}s (captura {cap:.2f}s) -> {len(toks)/total:.1f} tok/s ===", flush=True)
    print("FULL:" + "".join(full), flush=True)


if __name__ == "__main__":
    main()
