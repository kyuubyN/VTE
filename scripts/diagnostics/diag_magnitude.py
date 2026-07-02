import numpy as np, ctypes
from vte.core.model import VTEModel

model = VTEModel.from_pretrained('qwen2.5:1.5b-q4_k_m', use_hip_graph=False)
print('=== LOADED OK ===')

def read_tensor(name, n=1536):
    ptr = model.tensor_mapping.get(name)
    if ptr is None: return None
    pv = ptr.ptr if hasattr(ptr,'ptr') else ptr
    buf = bytearray(n*2)
    model._hip.safe_memcpy_device_to_host(buf, ctypes.c_void_p(pv), tag='output_debug')
    return np.frombuffer(bytes(buf), dtype=np.float16).astype(np.float32)

ids = model.tokenizer.encode('The capital of France is')
model.executor.prefill(ids)

print('input_embeddings: std=%.4f absmax=%.4f' % (
    (lambda a:(a.std(),np.abs(a).max()))(read_tensor('input_embeddings'))))
for l in [0,1,2,3,5,10,15,20,25,27]:
    a = read_tensor(f'blk.{l}.output')
    if a is not None:
        print('blk.%d.output: std=%.4f absmax=%.4f' % (l, a.std(), np.abs(a).max()))
o = read_tensor('output_norm.output')
print('output_norm.output: std=%.4f absmax=%.4f' % (o.std(), np.abs(o).max()))

# Detalhe da camada 0: sub-etapas
print()
print('--- Sub-etapas camada 0 ---')
for name, n in [('blk.0.attn_norm.output',1536),('blk.0.q_proj.output',1536),
                ('blk.0.attention.output',1536),('blk.0.attn_output.output',1536),
                ('blk.0.residual_1.output',1536),('blk.0.ffn_norm.output',1536),
                ('blk.0.gate_proj.output',8960),('blk.0.up_proj.output',8960),
                ('blk.0.swiglu.output',8960),('blk.0.down_proj.output',1536),
                ('blk.0.output',1536)]:
    a = read_tensor(name, n)
    if a is not None:
        print('%-28s std=%.4f absmax=%.4f' % (name, a.std(), np.abs(a).max()))

model.unload()
print('=== UNLOADED OK ===')
