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

steps = [
    ('input_embeddings', 1536),
    ('blk.0.attn_norm.output', 1536),
    ('blk.0.q_proj.output', 1536),
    ('blk.0.k_proj.output', 256),
    ('blk.0.v_proj.output', 256),
    ('blk.0.attention.output', 1536),
    ('blk.0.attn_output.output', 1536),
    ('blk.0.residual_1.output', 1536),
]

results = {}
for prompt in ['Ola', 'Batata frita internacional']:
    ids = model.tokenizer.encode(prompt)
    if model._use_hip_graph:
        model.executor.execute_prefill(ids)
    else:
        model.executor.prefill(ids)
    results[prompt] = {name: read_tensor(name, n) for name, n in steps}

p1, p2 = list(results.keys())
print(f'=== COMPARACAO {p1!r} vs {p2!r} ===')
for name, n in steps:
    diff = np.max(np.abs(results[p1][name] - results[p2][name]))
    print(f'{name:30s} max_abs_diff={diff:.6f}')

model.unload()
print('=== UNLOADED OK ===')
