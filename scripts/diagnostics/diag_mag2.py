import numpy as np, ctypes
from vte.core.model import VTEModel
model = VTEModel.from_pretrained('qwen2.5:1.5b-q4_k_m', use_hip_graph=False)
print('=== LOADED OK ===')
def rd(name,n=1536):
    p=model.tensor_mapping.get(name); pv=p.ptr if hasattr(p,'ptr') else p
    b=bytearray(n*2); model._hip.safe_memcpy_device_to_host(b,ctypes.c_void_p(pv),tag='output_debug')
    return np.frombuffer(bytes(b),dtype=np.float16).astype(np.float32)
ids=model.tokenizer.encode('The capital of France is')
model.executor.prefill(ids)
for l in [0,5,13,20,27]:
    a=rd(f'blk.{l}.output'); print('blk.%d.output std=%.3f absmax=%.3f naninf=%s'%(l,a.std(),np.abs(a).max(),(np.isnan(a).any() or np.isinf(a).any())))
o=rd('output_norm.output'); print('output_norm std=%.3f absmax=%.3f'%(o.std(),np.abs(o).max()))
# logits
hp=model.tensor_mapping.get('output_norm.output'); hv=hp.ptr if hasattr(hp,'ptr') else hp
lp=model.lm_head.compute_logits(hv,seq_len=1)
lb=bytearray(model.lm_head.vocab_size*2)
model._hip.safe_memcpy_device_to_host(lb,ctypes.c_void_p(lp),tag='logits_debug')
lo=np.frombuffer(bytes(lb),dtype=np.float16).astype(np.float32)
print('logits std=%.3f absmax=%.3f nan=%d inf=%d'%(lo.std(),np.abs(lo[np.isfinite(lo)]).max() if np.isfinite(lo).any() else -1, np.isnan(lo).sum(), np.isinf(lo).sum()))
fin=lo[np.isfinite(lo)]
top5=np.argsort(np.where(np.isfinite(lo),lo,-1e9))[-5:][::-1]
print('top5:', [(int(t),model.tokenizer.decode([int(t)]),round(float(lo[t]),2)) for t in top5])
model.unload()
print('=== UNLOADED OK ===')
