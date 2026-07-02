import numpy as np
import gguf
from gguf import GGUFReader
from vte.compiler.dequantizer import dequantize_q4_k

reader = GGUFReader('Model/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf')
t = next(x for x in reader.tensors if x.name=='blk.0.attn_q.weight')
raw = bytes(t.data.tobytes())
n = int(np.prod(t.shape))
ref = gguf.dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1)[:n]
mine = dequantize_q4_k(raw, n)

# Analisa so o primeiro bloco de 256
r0 = ref[:256]; m0 = mine[:256]
diff = np.abs(r0-m0)
print('Primeiro bloco de 256 - diffs por sub-bloco de 32:')
for sb in range(8):
    d = diff[sb*32:(sb+1)*32].max()
    print(f'  sub-bloco {sb} (elems {sb*32}-{sb*32+31}): maxdiff={d:.6f}')
# mostra onde diverge
idx = np.argmax(diff)
print(f'primeira grande div no bloco0 idx={idx}: ref={r0[idx]:.5f} mine={m0[idx]:.5f}')
print('ref[120:136]:', r0[120:136])
print('mine[120:136]:', m0[120:136])
