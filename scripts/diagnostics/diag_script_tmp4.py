import numpy as np, ctypes
from vte.core.model import VTEModel

model = VTEModel.from_pretrained('qwen2.5:1.5b-q4_k_m', use_hip_graph=False)
print('=== LOADED OK ===')

def resolve_ptr(name):
    ptr = model.tensor_mapping.get(name)
    return ptr.ptr if hasattr(ptr, 'ptr') else ptr

def read_tensor(name, n_elements, dtype=np.float16):
    ptr_val = resolve_ptr(name)
    buf = bytearray(n_elements * np.dtype(dtype).itemsize)
    model._hip.safe_memcpy_device_to_host(buf, ctypes.c_void_p(ptr_val), tag='output_debug')
    return np.frombuffer(bytes(buf), dtype=dtype).astype(np.float32)

results = {}
for token_id in [46, 58400]:
    print(f'\n--- token_id={token_id} (seq_len=1) ---')
    model.executor.prefill([token_id])
    emb = read_tensor('input_embeddings', 1536)
    norm_w = read_tensor('blk.0.attn_norm.weight', 1536)
    attn_norm_out = read_tensor('blk.0.attn_norm.output', 1536)
    print('emb first5:', emb[:5])
    print('norm_weight first5:', norm_w[:5])
    print('attn_norm.output first5:', attn_norm_out[:5])
    # calcula rmsnorm esperado localmente
    ms = np.mean(emb.astype(np.float64)**2)
    expected = (emb / np.sqrt(ms + 1e-5) * norm_w).astype(np.float32)
    print('esperado (calculado localmente) first5:', expected[:5])
    print('diff GPU vs esperado:', np.max(np.abs(attn_norm_out - expected)))
    results[token_id] = dict(emb=emb, attn_norm_out=attn_norm_out)

t1, t2 = list(results.keys())
print(f'\n=== COMPARACAO token {t1} vs {t2} ===')
print('emb diff:', np.max(np.abs(results[t1]['emb'] - results[t2]['emb'])))
print('attn_norm.output diff:', np.max(np.abs(results[t1]['attn_norm_out'] - results[t2]['attn_norm_out'])))

model.unload()
print('=== UNLOADED OK ===')
