"""
Fase 4 do plano Granite: teste de fogo de troca dinamica de modelo no MESMO
processo Python -- carrega Qwen, gera, descarrega, carrega Granite, gera,
descarrega de novo. Confirma que o hipFree do SlabAllocator e o remapeamento
do GGUF funcionam sem crash/vazamento entre modelos de arquiteturas
DIFERENTES (nao so tamanhos diferentes do mesmo modelo).
"""
from vte.core.model import VTEModel

VRAM_TOTAL_MB = 8176.0


def run_generation(model, prompt_text, n_tokens=10, apply_template=True):
    prompt = model.tokenizer.apply_chat_template(prompt_text) if apply_template else prompt_text
    out = ""
    for w in model.generate(prompt, max_tokens=n_tokens):
        out += w
    return out


print("=== 1) Carrega Qwen 1.5B ===")
qwen = VTEModel.from_pretrained("qwen2.5:1.5b-q4_k_m", context_length=512)
usage = qwen.get_vram_usage()
print(f"VRAM Qwen: {usage['total_mb']:.1f} MB ({usage['total_mb']/VRAM_TOTAL_MB*100:.1f}%)")
out = run_generation(qwen, "Quanto e 5 mais 7?")
print("Qwen gerou:", repr(out))

print("\n=== 2) Descarrega Qwen ===")
qwen.unload()
print("Qwen descarregado.")

print("\n=== 3) Carrega Granite 3B (arquitetura DIFERENTE, mesmo processo) ===")
granite = VTEModel.from_pretrained("granite-4.1:3b-q8_0", context_length=512)
usage = granite.get_vram_usage()
print(f"VRAM Granite: {usage['total_mb']:.1f} MB ({usage['total_mb']/VRAM_TOTAL_MB*100:.1f}%)")
out = run_generation(granite, "Quanto e 5 mais 7?")
print("Granite gerou:", repr(out))

print("\n=== 4) Descarrega Granite ===")
granite.unload()
print("Granite descarregado.")

print("\n=== 5) Recarrega Qwen de novo (confirma que o slot de VRAM foi liberado de verdade) ===")
qwen2 = VTEModel.from_pretrained("qwen2.5:1.5b-q4_k_m", context_length=512)
usage = qwen2.get_vram_usage()
print(f"VRAM Qwen (2a carga): {usage['total_mb']:.1f} MB ({usage['total_mb']/VRAM_TOTAL_MB*100:.1f}%)")
out = run_generation(qwen2, "Quanto e 5 mais 7?")
print("Qwen (2a carga) gerou:", repr(out))
qwen2.unload()

print("\n=== TESTE DE FOGO: PASSOU (troca dinamica Qwen <-> Granite sem crash) ===")
