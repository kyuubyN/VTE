# Teste de geracao com o modelo Qwen 3.5 usando os novos parametros defaults de producao.
# Este teste simula a resposta ao prompt do usuario para validar se o bloco de pensamento eh gerado de forma coerente e sem loops infinitos de sinonimos.

import sys
from vte.core.model import VTEModel

model = VTEModel.from_pretrained("qwen3.5:2b-q6_k")
model._lifecycle.ensure_loaded()

prompt = model.tokenizer.apply_chat_template(
    "Como fazer um hello world em Python?",
    enable_thinking=True
)

print("=== Gerando com defaults do Qwen 3.5 ===")
for word in model.generate(prompt, max_tokens=1000):
    print(word, end="", flush=True)
print("\n=== Fim da geracao ===")

model.unload()
