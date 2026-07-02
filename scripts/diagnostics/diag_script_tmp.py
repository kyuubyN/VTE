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
for prompt in ['Ola', 'Batata frita internacional']:
    ids = model.tokenizer.encode(prompt)
    print(f'\n--- prompt={prompt!r} ids={ids}')

    if model._use_hip_graph:
        model.executor.execute_prefill(ids)
    else:
        model.executor.prefill(ids)

    emb = read_tensor('input_embeddings', 1536)
    l0 = read_tensor('blk.0.output', 1536)
    l13 = read_tensor('blk.13.output', 1536)
    final = read_tensor('output_norm.output', 1536)

    logits_ptr = model.lm_head.compute_logits(resolve_ptr('output_norm.output'), seq_len=1)
    logits_buf = bytearray(model.lm_head.vocab_size * 4)
    model._hip.safe_memcpy_device_to_host(logits_buf, ctypes.c_void_p(logits_ptr), tag='logits_debug')
    logits = np.frombuffer(bytes(logits_buf), dtype=np.float32)
    top5 = np.argsort(logits)[-5:][::-1]

    print('input_embeddings first5:', emb[:5])
    print('blk.0.output   first5:', l0[:5])
    print('blk.13.output  first5:', l13[:5])
    print('output_norm.output first5:', final[:5])
    print('logits top5 ids:', top5, 'vals:', logits[top5])

    results[prompt] = dict(emb=emb, l0=l0, l13=l13, final=final, logits=logits, top5=top5)

p1, p2 = list(results.keys())
print(f'\n=== COMPARACAO {p1!r} vs {p2!r} ===')
for key in ['emb', 'l0', 'l13', 'final']:
    diff = np.max(np.abs(results[p1][key] - results[p2][key]))
    print(f'{key}: max_abs_diff entre prompts = {diff:.6f}')
print('logits top5 iguais?', np.array_equal(results[p1]['top5'], results[p2]['top5']))

model.unload()
print('=== UNLOADED OK ===')
