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

model.executor.prefill([46])
emb = read_tensor('input_embeddings', 1536)
norm_w = read_tensor('blk.0.attn_norm.weight', 1536)
gpu_out = read_tensor('blk.0.attn_norm.output', 1536)

# Referencia numpy padrao (RMSNorm classico)
ms = np.mean(emb.astype(np.float64)**2)
rms_inv = 1.0 / np.sqrt(ms + 1e-5)
ref = (emb.astype(np.float64) * rms_inv * norm_w.astype(np.float64)).astype(np.float32)

print('emb stats: mean=%.6f std=%.6f min=%.6f max=%.6f' % (emb.mean(), emb.std(), emb.min(), emb.max()))
print('mean_sq (ms):', ms, 'rms_inv:', rms_inv)
print('norm_w stats: mean=%.6f std=%.6f' % (norm_w.mean(), norm_w.std()))
print()
print('gpu_out[:10]:', gpu_out[:10])
print('ref[:10]:    ', ref[:10])
print('diff[:10]:   ', gpu_out[:10] - ref[:10])
print()
print('max abs diff geral:', np.max(np.abs(gpu_out - ref)))
print('indice do maior diff:', np.argmax(np.abs(gpu_out - ref)))
idx = np.argmax(np.abs(gpu_out - ref))
print(f'no indice {idx}: gpu={gpu_out[idx]} ref={ref[idx]} emb={emb[idx]} w={norm_w[idx]}')

# Testa hipotese: e se o kernel usar sum ao inves de mean (ou seja, dividir por 1 ao inves de 1536)?
ms_sum = np.sum(emb.astype(np.float64)**2)
rms_inv_sum = 1.0/np.sqrt(ms_sum+1e-5)
ref_sum = (emb.astype(np.float64)*rms_inv_sum*norm_w.astype(np.float64)).astype(np.float32)
print()
print('Hipotese SUM (sem dividir por hidden_size) diff:', np.max(np.abs(gpu_out-ref_sum)))

model.unload()
print('=== UNLOADED OK ===')
