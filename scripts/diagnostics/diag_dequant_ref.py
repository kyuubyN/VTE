import numpy as np
import gguf
from gguf import GGUFReader, dequantize
from vte.compiler.dequantizer import dequantize_q4_k, dequantize_q6_k
import math

reader = GGUFReader('Model/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf')

# pega tensores por tipo
targets = {}
for t in reader.tensors:
    if t.name == 'blk.0.attn_q.weight' and 'q4k' not in targets:
        targets['q4k'] = t
    if t.name == 'token_embd.weight' and 'q6k' not in targets:
        targets['q6k'] = t

for kind, t in targets.items():
    raw = bytes(t.data.tobytes()) if hasattr(t.data,'tobytes') else bytes(t.data)
    n = int(np.prod(t.shape))
    # referencia oficial
    ref = gguf.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1)[:n]
    # minha
    if kind == 'q4k':
        mine = dequantize_q4_k(raw, n)
    else:
        mine = dequantize_q6_k(raw, n)
    diff = np.max(np.abs(ref - mine))
    print(f'{kind} ({t.name}): n={n} shape={tuple(t.shape)} ttype={t.tensor_type}')
    print(f'  ref[:6]={ref[:6]}')
    print(f'  mine[:6]={mine[:6]}')
    print(f'  MAX DIFF = {diff:.6f}')
