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
ids = model.tokenizer.encode('The capital of France is')
S = len(ids)
print('S=',S,'ids=',ids)
model.executor.prefill(ids)

Wq=rd('blk.0.attn_q.weight',(NQ*HD,H)); bq=rd('blk.0.attn_q.bias',(NQ*HD,))
Wk=rd('blk.0.attn_k.weight',(NKV*HD,H)); bk=rd('blk.0.attn_k.bias',(NKV*HD,))
Wv=rd('blk.0.attn_v.weight',(NKV*HD,H)); bv=rd('blk.0.attn_v.bias',(NKV*HD,))
w_an=rd('blk.0.attn_norm.weight',(H,))
emb=rd('input_embeddings',(S,H))

def rms(x,w):
    ms=np.mean(x.astype(np.float64)**2,axis=-1,keepdims=True); return (x/np.sqrt(ms+eps)*w).astype(np.float32)

an=rms(emb,w_an)  # (S,H)
q=an@Wq.T+bq  # (S, 1536)
k=an@Wk.T+bk  # (S, 256)
v=an@Wv.T+bv

# RoPE: half=64, freqs
half=HD//2
freqs=1.0/(theta**(np.arange(0,half)/half))
def rope(x, nheads):
    x=x.reshape(S,nheads,HD).astype(np.float64)
    out=np.zeros_like(x)
    for p in range(S):
        ang=p*freqs; c=np.cos(ang); s=np.sin(ang)
        x1=x[p,:,:half]; x2=x[p,:,half:]
        out[p,:,:half]=x1*c - x2*s
        out[p,:,half:]=x1*s + x2*c
    return out.reshape(S,nheads*HD).astype(np.float32)
q=rope(q,NQ); k=rope(k,NKV)

# atencao causal GQA
qh=q.reshape(S,NQ,HD); kh=k.reshape(S,NKV,HD); vh=v.reshape(S,NKV,HD)
scale=1.0/np.sqrt(HD); g=NQ//NKV
attn=np.zeros((S,NQ,HD),dtype=np.float32)
for h in range(NQ):
    kvh=h//g
    for i in range(S):
        sc=np.array([ (qh[i,h]@kh[j,kvh])*scale for j in range(i+1)])
        sc=sc-sc.max(); w=np.exp(sc); w/=w.sum()
        attn[i,h]=sum(w[j]*vh[j,kvh] for j in range(i+1))
attn=attn.reshape(S,NQ*HD)

gpu_q=rd('blk.0.q_proj.output',(S,NQ*HD))
gpu_k=rd('blk.0.k_proj.output',(S,NKV*HD))
gpu_attn=rd('blk.0.attention.output',(S,NQ*HD))
print('q_proj (pre-rope no GPU?): diff vs ref-com-rope =', np.max(np.abs(gpu_q-q)))
print('  gpu_q[last,:4]=',gpu_q[-1,:4], 'ref_q[last,:4]=',q[-1,:4])
print('attention: diff =', np.max(np.abs(gpu_attn-attn)))
print('  gpu_attn[last,:4]=',gpu_attn[-1,:4],'ref[last,:4]=',attn[-1,:4])
model.unload()
print('=== UNLOADED OK ===')
