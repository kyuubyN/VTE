import numpy as np, ctypes
from vte.core.model import VTEModel

model = VTEModel.from_pretrained('qwen2.5:1.5b-q4_k_m', use_hip_graph=False)
print('=== LOADED OK ===')

def rd(name, shape):
    ptr = model.tensor_mapping.get(name)
    if ptr is None: return None
    pv = ptr.ptr if hasattr(ptr,'ptr') else ptr
    n = int(np.prod(shape))
    buf = bytearray(n*2)
    model._hip.safe_memcpy_device_to_host(buf, ctypes.c_void_p(pv), tag='output_debug')
    return np.frombuffer(bytes(buf), dtype=np.float16).astype(np.float32).reshape(shape)

# roda prefill de 1 token
tid = model.tokenizer.encode('The capital of France is')[0]
model.executor.prefill([tid])

H = 1536; HD = 128; NQ = 12; NKV = 2; FFN = 8960; eps = 1e-6

# Le pesos
Wq = rd('blk.0.attn_q.weight', (NQ*HD, H)); bq = rd('blk.0.attn_q.bias', (NQ*HD,))
Wk = rd('blk.0.attn_k.weight', (NKV*HD, H)); bk = rd('blk.0.attn_k.bias', (NKV*HD,))
Wv = rd('blk.0.attn_v.weight', (NKV*HD, H)); bv = rd('blk.0.attn_v.bias', (NKV*HD,))
Wo = rd('blk.0.attn_output.weight', (H, NQ*HD))
Wg = rd('blk.0.ffn_gate.weight', (FFN, H))
Wu = rd('blk.0.ffn_up.weight', (FFN, H))
Wd = rd('blk.0.ffn_down.weight', (H, FFN))
w_an = rd('blk.0.attn_norm.weight', (H,))
w_fn = rd('blk.0.ffn_norm.weight', (H,))

emb = rd('input_embeddings', (H,))

def rmsnorm(x, w):
    ms = np.mean(x.astype(np.float64)**2)
    return (x/np.sqrt(ms+eps)*w).astype(np.float32)

# Referencia SEM bias
an = rmsnorm(emb, w_an)
q_nobias = an @ Wq.T
q_bias = q_nobias + bq
# pos 0 -> RoPE identidade. attention 1 token -> out = v
v = an @ Wv.T + bv
# na verdade attention output tem NQ*HD dims; para 1 token = v repetido por grupo... simplificando: comparo so ate q/v proj

print('bq stats: absmax=%.4f mean=%.4f' % (np.abs(bq).max(), bq.mean()))
print('bk stats: absmax=%.4f' % np.abs(bk).max())
print('bv stats: absmax=%.4f' % np.abs(bv).max())

gpu_an = rd('blk.0.attn_norm.output', (H,))
gpu_q = rd('blk.0.q_proj.output', (NQ*HD,))
gpu_v = rd('blk.0.v_proj.output', (NKV*HD,))

print()
print('attn_norm: diff GPU vs ref =', np.max(np.abs(gpu_an - an)))
print('q_proj: diff GPU vs ref(sem bias) =', np.max(np.abs(gpu_q - q_nobias)))
print('q_proj: diff GPU vs ref(com bias) =', np.max(np.abs(gpu_q - q_bias)))
print('  gpu_q[:5]:', gpu_q[:5])
print('  ref_nobias[:5]:', q_nobias[:5])
print('  ref_bias[:5]:', q_bias[:5])

model.unload()
print('=== UNLOADED OK ===')
