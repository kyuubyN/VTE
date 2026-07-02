import numpy as np, ctypes
from vte.core.model import VTEModel

model = VTEModel.from_pretrained('qwen2.5:1.5b-q4_k_m', use_hip_graph=False)
print('=== LOADED OK ===')

def rd(name, shape):
    ptr = model.tensor_mapping.get(name)
    if ptr is None: return None
    pv = ptr.ptr if hasattr(ptr,'ptr') else ptr
    n=int(np.prod(shape)); buf=bytearray(n*2)
    model._hip.safe_memcpy_device_to_host(buf, ctypes.c_void_p(pv), tag='output_debug')
    return np.frombuffer(bytes(buf),dtype=np.float16).astype(np.float32).reshape(shape)

H=1536;HD=128;NQ=12;NKV=2;FFN=8960;eps=1e-6;NL=28
tid = model.tokenizer.encode('The capital of France is')[0]
model.executor.prefill([tid])

def rms(x,w):
    ms=np.mean(x.astype(np.float64)**2); return (x/np.sqrt(ms+eps)*w).astype(np.float32)

x = rd('input_embeddings',(H,))
for l in range(NL):
    Wq=rd(f'blk.{l}.attn_q.weight',(NQ*HD,H)); bq=rd(f'blk.{l}.attn_q.bias',(NQ*HD,))
    Wk=rd(f'blk.{l}.attn_k.weight',(NKV*HD,H)); bk=rd(f'blk.{l}.attn_k.bias',(NKV*HD,))
    Wv=rd(f'blk.{l}.attn_v.weight',(NKV*HD,H)); bv=rd(f'blk.{l}.attn_v.bias',(NKV*HD,))
    Wo=rd(f'blk.{l}.attn_output.weight',(H,NQ*HD))
    Wg=rd(f'blk.{l}.ffn_gate.weight',(FFN,H)); Wu=rd(f'blk.{l}.ffn_up.weight',(FFN,H)); Wd=rd(f'blk.{l}.ffn_down.weight',(H,FFN))
    w_an=rd(f'blk.{l}.attn_norm.weight',(H,)); w_fn=rd(f'blk.{l}.ffn_norm.weight',(H,))
    an=rms(x,w_an); v=an@Wv.T+bv
    attn=np.zeros(NQ*HD,dtype=np.float32); g=NQ//NKV
    for h in range(NQ): attn[h*HD:(h+1)*HD]=v[(h//g)*HD:(h//g+1)*HD]
    res1=x+attn@Wo.T
    fn=rms(res1,w_fn); gate=fn@Wg.T; up=fn@Wu.T
    silu=gate/(1.0+np.exp(-gate)); down=(silu*up)@Wd.T
    x=res1+down

w_on=rd('output_norm.weight',(H,))
ref_final=rms(x,w_on)
gpu_final=rd('output_norm.output',(H,))
print('output_norm final: diff=%.4f ref_absmax=%.3f gpu_absmax=%.3f' % (
    np.max(np.abs(ref_final-gpu_final)), np.abs(ref_final).max(), np.abs(gpu_final).max()))

# logits ref (tied embedding)
emb_w=rd('token_embd.weight',(151936,H))
ref_logits=ref_final@emb_w.T
top5=np.argsort(ref_logits)[-5:][::-1]
print('REF top5:', [(int(t), model.tokenizer.decode([int(t)]), round(float(ref_logits[t]),2)) for t in top5])

model.unload()
print('=== UNLOADED OK ===')
