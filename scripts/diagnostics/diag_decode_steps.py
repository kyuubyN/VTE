import numpy as np, ctypes
from vte.core.model import VTEModel

model = VTEModel.from_pretrained('qwen2.5:1.5b-q4_k_m', use_hip_graph=False)
print('=== LOADED OK ===')

prompt = 'The capital of France is'
ids = model.tokenizer.encode(prompt)
print('prompt ids:', ids)

model.executor.prefill(ids)
current = len(ids)

def get_logits():
    hp = model.tensor_mapping.get('output_norm.output')
    hv = hp.ptr if hasattr(hp,'ptr') else hp
    lp = model.lm_head.compute_logits(hv, seq_len=1)
    buf = bytearray(model.lm_head.vocab_size*4)
    model._hip.safe_memcpy_device_to_host(buf, ctypes.c_void_p(lp), tag='logits_debug')
    return np.frombuffer(bytes(buf), dtype=np.float32)

tokens = list(ids)
for step in range(6):
    model.executor.decode_step(tokens[-1], current)
    logits = get_logits()
    top5 = np.argsort(logits)[-5:][::-1]
    decoded = [model.tokenizer.decode([int(t)]) for t in top5]
    print(f'step {step} pos={current}: logits mean={logits.mean():.3f} std={logits.std():.3f} top5={list(zip([int(t) for t in top5], decoded, [round(float(logits[t]),2) for t in top5]))}')
    nxt = int(top5[0])  # greedy para diagnostico
    tokens.append(nxt)
    current += 1

model.unload()
print('=== UNLOADED OK ===')
