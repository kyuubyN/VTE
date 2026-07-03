"""
Fase 3 do plano Granite: teste de estresse de VRAM com geracao longa.

Mede o footprint real (pesos + KV cache + arena + scratch) ao vivo durante
uma geracao de centenas de tokens, e confirma que o Memory Guardian
intervem graciosamente (HIPSafetyError, nao crash) quando o KV cache
pedido excede a VRAM disponivel.
"""
import time
from vte.core.model import VTEModel
from vte.bridge.errors import HIPSafetyError

VRAM_TOTAL_MB = 8176.0  # RX 7600, medido nesta sessao

print("=== Parte 1: geracao longa (500 tokens), monitorando VRAM ===")
m = VTEModel.from_pretrained("granite-4.1:3b-q8_0", context_length=1024)

usage0 = m.get_vram_usage()
print(f"VRAM apos load: {usage0['total_mb']:.1f} MB ({usage0['total_mb']/VRAM_TOTAL_MB*100:.1f}% de {VRAM_TOTAL_MB} MB)")
print(f"  pesos={usage0['weights_mb']:.1f} MB kv_cache={usage0['kv_cache_mb']:.1f} MB arena={usage0['arena_mb']:.1f} MB scratch={usage0['scratch_mb']:.1f} MB")

prompt = m.tokenizer.apply_chat_template(
    "Escreva um texto longo e detalhado, com pelo menos 500 palavras, sobre a historia da exploracao espacial."
)

t0 = time.time()
n_tokens = 0
out_chars = 0
for w in m.generate(prompt, max_tokens=500, temperature=0.7, top_p=0.9):
    n_tokens += 1
    out_chars += len(w)
    if n_tokens % 100 == 0:
        elapsed = time.time() - t0
        usage = m.get_vram_usage()
        print(f"  [{n_tokens} tokens, {elapsed:.1f}s, {n_tokens/elapsed:.1f} tok/s] "
              f"VRAM={usage['total_mb']:.1f} MB ({usage['total_mb']/VRAM_TOTAL_MB*100:.1f}%)")

elapsed = time.time() - t0
usage_final = m.get_vram_usage()
print(f"\nGeracao completa: {n_tokens} tokens em {elapsed:.1f}s ({n_tokens/max(elapsed,0.001):.1f} tok/s)")
print(f"VRAM final: {usage_final['total_mb']:.1f} MB ({usage_final['total_mb']/VRAM_TOTAL_MB*100:.1f}%)")
print(f"Texto gerado tem {out_chars} caracteres (sanidade: >0 e sem crash = OK)")

m.unload()
print("Modelo descarregado com sucesso.\n")

print("=== Parte 2: guard de VRAM sob pressao deliberada (context_length absurdo) ===")
print("Tentando context_length=131072 (nativo do Granite) -- KV cache sozinho excederia a VRAM disponivel.")
try:
    m2 = VTEModel.from_pretrained("granite-4.1:3b-q8_0", context_length=131072)
    print("INESPERADO: carregou sem erro (VRAM deve ter sobrado mais do que o previsto).")
    print("VRAM:", m2.get_vram_usage())
    m2.unload()
except HIPSafetyError as e:
    print(f"OK -- Memory Guardian interveio graciosamente (HIPSafetyError, sem crash):\n  {e}")
except Exception as e:
    print(f"FALHA -- excecao inesperada (nao HIPSafetyError): {type(e).__name__}: {e}")
