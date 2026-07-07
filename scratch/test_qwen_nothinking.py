# Teste de geracao do Qwen 3.5 com o thinking mode desativado no template de chat.
# Este teste valida se o modelo vai direto para a resposta correta e nao entra em colapso.

import sys
from vte.core.model import VTEModel

model = VTEModel.from_pretrained("qwen3.5:2b-q6_k")
model._lifecycle.ensure_loaded()

prompt = model.tokenizer.apply_chat_template(
    "Como fazer um hello world em Python?",
    enable_thinking=False
)

print("=== Gerando sem thinking mode ===")
for word in model.generate(prompt, max_tokens=600):
    print(word, end="", flush=True)
print("\n=== Fim da geracao ===")

model.unload()
