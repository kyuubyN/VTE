"""Teste end-to-end da API de producao generate_batch (Fase II prep)."""
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
import sys
sys.path.insert(0, os.path.abspath("."))
from vte.core.model import VTEModel

model = VTEModel.from_pretrained("qwen2.5:1.5b-q4_k_m", use_hip_graph=True, idle_timeout_seconds=120, max_batch_size=4)

prompts = [
    "The capital of France is",
    "Once upon a time, a knight",
    "2 + 2 equals",
    "My favorite color is",
]
# Garante mesmo comprimento de tokens (requisito desta etapa).
lens = [len(model.tokenizer.encode(p)) for p in prompts]
print("Comprimentos dos prompts (tokens):", lens)

max_len = max(lens)
padded_prompts = []
for p, l in zip(prompts, lens):
    # Padding TRIVIAL só para bater o requisito de mesmo comprimento neste
    # teste (repete o último token do prompt) — não é uma solução de
    # mascaramento correta, só serve para exercitar a API com textos reais.
    if l < max_len:
        pass  # todos os prompts de teste já tem tamanhos parecidos; ajusta abaixo se necessario
    padded_prompts.append(p)

if len(set(lens)) != 1:
    print("Prompts com tamanhos diferentes; usando os primeiros N com mesmo tamanho ou ajustando manualmente.")
    # Agrupa por tamanho e pega o maior grupo
    from collections import Counter
    common_len = Counter(lens).most_common(1)[0][0]
    prompts = [p for p, l in zip(prompts, lens) if l == common_len]
    while len(prompts) < 4:
        prompts.append(prompts[0])
    prompts = prompts[:4]
    print("Prompts ajustados:", prompts)

outputs = [[] for _ in prompts]
for words in model.generate_batch(prompts, max_tokens=40, temperature=0.7):
    for b, w in enumerate(words):
        outputs[b].append(w)

for i, (p, out) in enumerate(zip(prompts, outputs)):
    print(f"\n--- Sequencia {i} ---")
    print("Prompt:", p)
    print("Geracao:", "".join(out))
