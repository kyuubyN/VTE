import numpy as np, ctypes
from vte.core.model import VTEModel
model = VTEModel.from_pretrained('qwen2.5:1.5b-q4_k_m', use_hip_graph=False)
print('=== LOADED OK ===')
def rd(name, shape):
    p=model.tensor_mapping.get(name); pv=p.ptr if hasattr(p,'ptr') else p
    n=int(np.prod(shape)); b=bytearray(n*2)
    model._hip.safe_memcpy_device_to_host(b,ctypes.c_void_p(pv),tag='output_debug')
    return np.frombuffer(bytes(b),dtype=np.float16).astype(np.float32).reshape(shape)
H=1536;HD=128;NQ=12;NKV=2;eps=1e-6;theta=1000000.0
ids=model.tokenizer.encode('The capital of France is'); S=len(ids)
model.executor.prefill(ids)
Wq=rd('blk.0.attn_q.weight',(NQ*HD,H)); bq=rd('blk.0.attn_q.bias',(NQ*HD,))
w_an=rd('blk.0.attn_norm.weight',(H,)); emb=rd('input_embeddings',(S,H))
def rms(x,w):
    ms=np.mean(x.astype(np.float64)**2,axis=-1,keepdims=True); return (x/np.sqrt(ms+eps)*w).astype(np.float32)
an=rms(emb,w_an); q_norope=an@Wq.T+bq
half=HD//2; freqs=1.0/(theta**(np.arange(0,half)/half))
def rope(x):
    x=x.reshape(S,NQ,HD).astype(np.float64); out=np.zeros_like(x)
    for p in range(S):
        ang=p*freqs; c=np.cos(ang); s=np.sin(ang)
        out[p,:,:half]=x[p,:,:half]*c - x[p,:,half:]*s
        out[p,:,half:]=x[p,:,:half]*s + x[p,:,half:]*c
    return out.reshape(S,NQ*HD).astype(np.float32)
q_rope=rope(q_norope)
gpu_q=rd('blk.0.q_proj.output',(S,NQ*HD))
for pos in range(S):
    dn=np.median(np.abs(gpu_q[pos]-q_norope[pos]))
    dr=np.median(np.abs(gpu_q[pos]-q_rope[pos]))
    print(f'pos {pos}: median|gpu-norope|={dn:.5f}  median|gpu-rope|={dr:.5f}')
model.unload()
print('=== UNLOADED OK ===')
