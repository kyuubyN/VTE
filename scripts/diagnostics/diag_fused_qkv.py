import numpy as np, ctypes
from vte.bridge.hip_runtime import HIPRuntime
from vte.bridge.memory import SlabAllocator, MemoryRegion
from vte.compiler.codegen import CodegenEngine
from vte.compiler.rope_cache_builder import RoPECacheBuilder

hip = HIPRuntime(); hip.initialize()
allocator = SlabAllocator(hip, 32*1024*1024); allocator.initialize()

H=1536; HD=128; NQ=12; NKV=2; eps=1e-6; theta=1000000.0
np.random.seed(5)
x = (np.random.randn(H)*0.3).astype(np.float16)
w_an = (np.random.randn(H)*0.2+1.0).astype(np.float16)
Wq = (np.random.randn(NQ*HD,H)*0.05).astype(np.float16); bq=(np.random.randn(NQ*HD)*0.1).astype(np.float16)
Wk = (np.random.randn(NKV*HD,H)*0.05).astype(np.float16); bk=(np.random.randn(NKV*HD)*0.1).astype(np.float16)
Wv = (np.random.randn(NKV*HD,H)*0.05).astype(np.float16); bv=(np.random.randn(NKV*HD)*0.1).astype(np.float16)

builder = RoPECacheBuilder(max_seq_len=32, head_dim=HD, rope_theta=theta)
cos_cache, sin_cache = builder.build_cache()
cos_ptr, sin_ptr = builder.upload_to_vram(cos_cache, sin_cache, hip, allocator)

def up(arr,tag):
    b=allocator.allocate(len(arr.tobytes()),tag,MemoryRegion.SCRATCH)
    hip.safe_memcpy_host_to_device(ctypes.c_void_p(b.ptr),arr.tobytes(),tag=tag); return b

xb=up(x,'x'); wanb=up(w_an,'wan')
wqb=up(Wq.reshape(-1),'wq'); bqb=up(bq,'bq')
wkb=up(Wk.reshape(-1),'wk'); bkb=up(bk,'bk')
wvb=up(Wv.reshape(-1),'wv'); bvb=up(bv,'bv')
offb=allocator.allocate(4,'off',MemoryRegion.SCRATCH)
POS=7
hip.safe_memcpy_host_to_device(ctypes.c_void_p(offb.ptr),np.array([POS],dtype=np.int32).tobytes(),tag='off')

eng=CodegenEngine()
hsaco=eng.compile_kernel('fused_norm_matmul_rope',arch=hip.get_gpu_architecture())
mod,fn=hip.load_kernel(hsaco,'fused_norm_matmul_rope_kernel')

def run(wb,bb,out_features,apply_rope):
    ob=allocator.allocate(out_features*2,'o',MemoryRegion.SCRATCH)
    args=[ctypes.c_void_p(xb.ptr),ctypes.c_void_p(wanb.ptr),ctypes.c_void_p(wb.ptr),
          ctypes.c_void_p(bb.ptr),ctypes.c_void_p(cos_ptr),ctypes.c_void_p(sin_ptr),
          ctypes.c_void_p(ob.ptr),ctypes.c_int(H),ctypes.c_int(out_features),ctypes.c_int(HD),
          ctypes.c_float(eps),ctypes.c_void_p(offb.ptr),ctypes.c_int(1 if apply_rope else 0)]
    grid=(out_features//HD,1,1)
    hip.launch_kernel(function=fn,grid=grid,block=(256,1,1),args=args,shared_mem=0,expected_args=13)
    hip.synchronize()
    buf=bytearray(out_features*2)
    hip.safe_memcpy_device_to_host(buf,ctypes.c_void_p(ob.ptr),tag='output_debug')
    return np.frombuffer(bytes(buf),dtype=np.float16).astype(np.float32)

gpu_q = run(wqb,bqb,NQ*HD,True)
gpu_k = run(wkb,bkb,NKV*HD,True)
gpu_v = run(wvb,bvb,NKV*HD,False)

# referencia numpy
def rms(xx,ww,eps):
    ms=np.mean(xx.astype(np.float64)**2); return (xx/np.sqrt(ms+eps)*ww).astype(np.float32)
xn = rms(x.astype(np.float32), w_an.astype(np.float32), eps)
q = xn@Wq.astype(np.float32).T + bq.astype(np.float32)
k = xn@Wk.astype(np.float32).T + bk.astype(np.float32)
v = xn@Wv.astype(np.float32).T + bv.astype(np.float32)
half=HD//2
freqs=1.0/(theta**(np.arange(0,half)/half))
def rope(vec,nheads,pos):
    vv=vec.reshape(nheads,HD).astype(np.float64); out=np.zeros_like(vv)
    ang=pos*freqs; c=np.cos(ang); s=np.sin(ang)
    out[:,:half]=vv[:,:half]*c - vv[:,half:]*s
    out[:,half:]=vv[:,:half]*s + vv[:,half:]*c
    return out.reshape(-1).astype(np.float32)
ref_q = rope(q,NQ,POS); ref_k = rope(k,NKV,POS); ref_v = v

dq,dk,dv = np.max(np.abs(gpu_q-ref_q)), np.max(np.abs(gpu_k-ref_k)), np.max(np.abs(gpu_v-ref_v))
print('Q diff:', dq)
print('K diff:', dk)
print('V diff:', dv)
print('gpu_q[:4]=',gpu_q[:4],'ref_q[:4]=',ref_q[:4])
worst=max(dq,dk,dv)
print(f'PIOR diff = {worst:.6f}  ({"PASS" if worst<2e-3 else "FAIL"} @ tol 2e-3)')

import subprocess, os
rb=r"C:\Program Files\AMD\ROCm\6.4\bin"
raw=os.path.join(os.environ.get("TEMP","/tmp"),"fused_qkv_gfx1102.o")
subprocess.run([os.path.join(rb,"clang-offload-bundler.exe"),"--type=o","--unbundle",
  f"--input={hsaco}","--targets=hipv4-amdgcn-amd-amdhsa--gfx1102",f"--output={raw}"],capture_output=True)
out=subprocess.run([os.path.join(rb,"llvm-objdump.exe"),"-d",raw],capture_output=True,text=True).stdout
print(f"ISA: global_load_b128={out.count('global_load_b128')}  "
      f"ds_swizzle/bpermute={out.count('ds_swizzle')+out.count('ds_bpermute')+out.count('ds_permute')}  "
      f"VETORIZACAO={'CONFIRMADA' if out.count('global_load_b128')>0 else 'FALHOU'}")
hip.cleanup()
