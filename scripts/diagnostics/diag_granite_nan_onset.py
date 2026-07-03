"""
Localiza a posicao EXATA onde o hidden state/logits do Granite comecam a
virar NaN durante uma geracao longa -- em vez de so esperar o crash do
sampler, inspeciona os logits token a token.
"""
import numpy as np
import ctypes
from vte.core.model import VTEModel

m = VTEModel.from_pretrained("granite-4.1:3b-q8_0", context_length=512)
prompt = m.tokenizer.apply_chat_template(
    "Escreva um texto longo sobre a historia da exploracao espacial."
)
tokens = m.tokenizer.encode(prompt)
print(f"prompt tem {len(tokens)} tokens")

vocab_size = m.lm_head.vocab_size


def read_logits():
    buf = bytearray(vocab_size * 2)
    m._hip.safe_memcpy_device_to_host(buf, ctypes.c_void_p(m.lm_head.logits_buffer), tag="logits")
    return np.frombuffer(bytes(buf), dtype=np.float16).astype(np.float32)


def read_hidden():
    ptr = m.tensor_mapping.get("output_norm.output")
    val = ptr.ptr if hasattr(ptr, "ptr") else ptr
    H = m.metadata["embedding_length"]
    buf = bytearray(H * 2)
    m._hip.safe_memcpy_device_to_host(buf, ctypes.c_void_p(val), tag="output")
    return np.frombuffer(bytes(buf), dtype=np.float16).astype(np.float32)


# Prefill
for pos, tok in enumerate(tokens):
    m.executor.execute_decode(tok, kv_offset=pos)
m._hip.synchronize()

current_seq_len = len(tokens)
generated_text = []
first_nan_pos = None

np.random.seed(0)
for step in range(500):
    logits = read_logits()
    hidden = read_hidden()
    if np.isnan(logits).any() or np.isnan(hidden).any():
        first_nan_pos = current_seq_len
        print(f"\n!!! NaN detectado na posicao (seq_len) {current_seq_len} (step {step}) !!!")
        print("hidden nan?", np.isnan(hidden).any(), "logits nan?", np.isnan(logits).any())
        print("Ultimos 15 tokens de texto antes do NaN:", repr("".join(generated_text[-15:])))
        break

    # sample (greedy simples para reprodutibilidade, sem depender do Sampler)
    next_token = int(np.argmax(logits))
    word = m.tokenizer.decode([next_token])
    generated_text.append(word)

    if next_token in m.tokenizer.stop_token_ids:
        print(f"EOS na posicao {current_seq_len}, sem NaN.")
        break

    current_seq_len += 1
    m.executor.execute_decode(next_token, kv_offset=current_seq_len - 1)
    m._hip.synchronize()

    if step % 50 == 0:
        print(f"[step {step}, seq_len {current_seq_len}] ok, ultimo token: {word!r}")

if first_nan_pos is None:
    print("\nNenhum NaN em 500 passos (greedy). Texto final:")
    print(repr("".join(generated_text)))
