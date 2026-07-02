import numpy as np, ctypes
from vte.bridge.hip_runtime import HIPRuntime
from vte.bridge.memory import SlabAllocator, MemoryRegion
from vte.compiler.codegen import CodegenEngine

hip = HIPRuntime(); hip.initialize()
allocator = SlabAllocator(hip, 32*1024*1024); allocator.initialize()

HD=128; NQ=12; NKV=2; scale=1.0/np.sqrt(HD)
np.random.seed(0)
# Simula um decode na posicao P: cache ja tem P posicoes + escrevemos a atual
P = 7  # kv_offset (token atual na posicao 7, atende a 0..7)
max_seq = 32

# k/v do token atual [1, NKV, HD]
k_cur = (np.random.randn(NKV*HD)*0.5).astype(np.float16)
v_cur = (np.random.randn(NKV*HD)*0.5).astype(np.float16)
q_cur = (np.random.randn(NQ*HD)*0.5).astype(np.float16)
# cache pre-preenchido com posicoes 0..P-1
kcache = (np.random.randn(max_seq*NKV*HD)*0.5).astype(np.float16)
vcache = (np.random.randn(max_seq*NKV*HD)*0.5).astype(np.float16)

# aloca
def up(arr,tag):
    b=allocator.allocate(len(arr)*2,tag,MemoryRegion.SCRATCH)
    hip.safe_memcpy_host_to_device(ctypes.c_void_p(b.ptr),arr.tobytes(),tag=tag); return b
qb=up(q_cur,'q'); kb=up(k_cur,'k'); vb=up(v_cur,'v')
kcb=up(kcache,'kc'); vcb=up(vcache,'vc')
ob=allocator.allocate(NQ*HD*2,'o',MemoryRegion.SCRATCH)
offb=allocator.allocate(4,'off',MemoryRegion.SCRATCH)
hip.safe_memcpy_host_to_device(ctypes.c_void_p(offb.ptr),np.array([P],dtype=np.int32).tobytes(),tag='off')

eng=CodegenEngine()
hsaco=eng.compile_kernel('flash_attention',arch=hip.get_gpu_architecture())
mod,fn=hip.load_kernel(hsaco,'flash_attention_kernel')
args=[ctypes.c_void_p(qb.ptr),ctypes.c_void_p(kb.ptr),ctypes.c_void_p(vb.ptr),
      ctypes.c_void_p(kcb.ptr),ctypes.c_void_p(vcb.ptr),ctypes.c_void_p(ob.ptr),
      ctypes.c_int(1),ctypes.c_int(NQ),ctypes.c_int(NKV),ctypes.c_int(HD),
      ctypes.c_float(scale),ctypes.c_void_p(offb.ptr)]
shared=2*HD*4
hip.launch_kernel(function=fn,grid=(1,NQ,1),block=(HD,1,1),args=args,shared_mem=shared,expected_args=12)
hip.synchronize()
outb=bytearray(NQ*HD*2)
hip.safe_memcpy_device_to_host(outb,ctypes.c_void_p(ob.ptr),tag='output_debug')
gpu=np.frombuffer(bytes(outb),dtype=np.float16).astype(np.float32).reshape(NQ,HD)

# referencia numpy: escreve k/v atual no cache pos P, depois atencao
kc=kcache.astype(np.float32).reshape(max_seq,NKV,HD).copy()
vc=vcache.astype(np.float32).reshape(max_seq,NKV,HD).copy()
kc[P]=k_cur.astype(np.float32).reshape(NKV,HD)
vc[P]=v_cur.astype(np.float32).reshape(NKV,HD)
q=q_cur.astype(np.float32).reshape(NQ,HD)
g=NQ//NKV
ref=np.zeros((NQ,HD),dtype=np.float32)
for h in range(NQ):
    kvh=h//g
    sc=np.array([ (q[h]@kc[j,kvh])*scale for j in range(P+1)])
    sc-=sc.max(); w=np.exp(sc); w/=w.sum()
    ref[h]=sum(w[j]*vc[j,kvh] for j in range(P+1))
print('max abs diff:', np.max(np.abs(gpu-ref)))
print('gpu[0,:4]=',gpu[0,:4],' ref[0,:4]=',ref[0,:4])
print('gpu[11,:4]=',gpu[11,:4],' ref[11,:4]=',ref[11,:4])
hip.cleanup()
