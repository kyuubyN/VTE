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

executor = model.executor
executor.context.rollback()
executor._write_input_ids([46])

print('--- Rodando SO a camada 0 ---')
executor.execute_layer(0, seq_len=1, kv_cache_offset=0)

emb = read_tensor('input_embeddings', 1536)
norm_w = read_tensor('blk.0.attn_norm.weight', 1536)
attn_norm_out = read_tensor('blk.0.attn_norm.output', 1536)

ms = np.mean(emb.astype(np.float64)**2)
rms_inv = 1.0/np.sqrt(ms+1e-5)
ref = (emb.astype(np.float64)*rms_inv*norm_w.astype(np.float64)).astype(np.float32)

print('attn_norm.output[:10] (apos SO layer 0):', attn_norm_out[:10])
print('ref[:10]:', ref[:10])
print('diff apos SO camada 0:', np.max(np.abs(attn_norm_out - ref)))

model.unload()
print('=== UNLOADED OK ===')
