import numpy as np, ctypes
from vte.core.model import VTEModel

model = VTEModel.from_pretrained('qwen2.5:1.5b-q4_k_m', use_hip_graph=False)
print('=== LOADED OK ===')

def rd(name, shape):
    ptr = model.tensor_mapping.get(name)
    if ptr is None: return None
    pv = ptr.ptr if hasattr(ptr,'ptr') else ptr
    n = int(np.prod(shape))
    buf = bytearray(n*2)
    model._hip.safe_memcpy_device_to_host(buf, ctypes.c_void_p(pv), tag='output_debug')
    return np.frombuffer(bytes(buf), dtype=np.float16).astype(np.float32).reshape(shape)

tid = model.tokenizer.encode('The capital of France is')[0]
model.executor.prefill([tid])

H=1536; HD=128; NQ=12; NKV=2; FFN=8960; eps=1e-6
Wq=rd('blk.0.attn_q.weight',(NQ*HD,H)); bq=rd('blk.0.attn_q.bias',(NQ*HD,))
Wk=rd('blk.0.attn_k.weight',(NKV*HD,H)); bk=rd('blk.0.attn_k.bias',(NKV*HD,))
Wv=rd('blk.0.attn_v.weight',(NKV*HD,H)); bv=rd('blk.0.attn_v.bias',(NKV*HD,))
Wo=rd('blk.0.attn_output.weight',(H,NQ*HD))
Wg=rd('blk.0.ffn_gate.weight',(FFN,H)); Wu=rd('blk.0.ffn_up.weight',(FFN,H)); Wd=rd('blk.0.ffn_down.weight',(H,FFN))
w_an=rd('blk.0.attn_norm.weight',(H,)); w_fn=rd('blk.0.ffn_norm.weight',(H,))
emb=rd('input_embeddings',(H,))

def rms(x,w):
    ms=np.mean(x.astype(np.float64)**2); return (x/np.sqrt(ms+eps)*w).astype(np.float32)

an=rms(emb,w_an)
q=an@Wq.T+bq; k=an@Wk.T+bk; v=an@Wv.T+bv
# RoPE pos0 identidade. Attention GQA 1 token: cada q-head copia v da kv-head correspondente
attn=np.zeros(NQ*HD, dtype=np.float32)
gqa=NQ//NKV
for h in range(NQ):
    kvh=h//gqa
    attn[h*HD:(h+1)*HD]=v[kvh*HD:(kvh+1)*HD]
attn_out=attn@Wo.T
res1=emb+attn_out
fn=rms(res1,w_fn)
gate=fn@Wg.T; up=fn@Wu.T
silu=gate/(1.0+np.exp(-gate)); swi=silu*up
down=swi@Wd.T
out=res1+down

def cmp(name, ref, gpu_name, shape):
    g=rd(gpu_name,shape)
    if g is None:
        print(f'{name}: GPU tensor ausente'); return
    d=np.max(np.abs(g.reshape(-1)-ref.reshape(-1)))
    print(f'{name:22s} diff={d:.4f}  ref_absmax={np.abs(ref).max():.3f} gpu_absmax={np.abs(g).max():.3f}')

cmp('attn_norm', an, 'blk.0.attn_norm.output',(H,))
cmp('q_proj', q, 'blk.0.q_proj.output',(NQ*HD,))
cmp('k_proj', k, 'blk.0.k_proj.output',(NKV*HD,))
cmp('v_proj', v, 'blk.0.v_proj.output',(NKV*HD,))
cmp('attention', attn, 'blk.0.attention.output',(NQ*HD,))
cmp('attn_output', attn_out, 'blk.0.attn_output.output',(H,))
cmp('residual_1', res1, 'blk.0.residual_1.output',(H,))
cmp('ffn_norm', fn, 'blk.0.ffn_norm.output',(H,))
cmp('gate_proj', gate, 'blk.0.gate_proj.output',(FFN,))
cmp('up_proj', up, 'blk.0.up_proj.output',(FFN,))
cmp('swiglu', swi, 'blk.0.swiglu.output',(FFN,))
cmp('down_proj', down, 'blk.0.down_proj.output',(H,))
cmp('blk.0.output', out, 'blk.0.output',(H,))

model.unload()
print('=== UNLOADED OK ===')
