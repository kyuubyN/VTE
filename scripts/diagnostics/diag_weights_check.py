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

norm_w = read_tensor('blk.0.attn_norm.weight', 1536)
print('ANTES do prefill - norm_w stats: mean=%.6f std=%.6f first5=%s' % (norm_w.mean(), norm_w.std(), norm_w[:5]))

model.executor.prefill([46])

norm_w2 = read_tensor('blk.0.attn_norm.weight', 1536)
print('DEPOIS do prefill - norm_w stats: mean=%.6f std=%.6f first5=%s' % (norm_w2.mean(), norm_w2.std(), norm_w2[:5]))

emb = read_tensor('input_embeddings', 1536)
print('DEPOIS do prefill - emb stats: mean=%.6f std=%.6f first5=%s' % (emb.mean(), emb.std(), emb[:5]))

model.unload()
print('=== UNLOADED OK ===')
