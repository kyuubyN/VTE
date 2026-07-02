import numpy as np, ctypes
from vte.bridge.hip_runtime import HIPRuntime
from vte.bridge.memory import SlabAllocator, MemoryRegion
from vte.compiler.codegen import CodegenEngine

hip = HIPRuntime(); hip.initialize()
allocator = SlabAllocator(hip, 256*1024*1024); allocator.initialize()

H=1536; FFN=8960; eps=1e-6
np.random.seed(9)
x = (np.random.randn(H)*0.3).astype(np.float16)
w_fn = (np.random.randn(H)*0.2+1.0).astype(np.float16)
Wg = (np.random.randn(FFN,H)*0.05).astype(np.float16)
Wu = (np.random.randn(FFN,H)*0.05).astype(np.float16)

def up(arr,tag):
    b=allocator.allocate(len(arr.tobytes()),tag,MemoryRegion.SCRATCH)
    hip.safe_memcpy_host_to_device(ctypes.c_void_p(b.ptr),arr.tobytes(),tag=tag); return b

xb=up(x,'x'); wfnb=up(w_fn,'wfn'); wgb=up(Wg.reshape(-1),'wg'); wub=up(Wu.reshape(-1),'wu')
ob=allocator.allocate(FFN*2,'o',MemoryRegion.SCRATCH)

eng=CodegenEngine()
hsaco=eng.compile_kernel('fused_rmsnorm_gate_up_silu',arch=hip.get_gpu_architecture())
mod,fn=hip.load_kernel(hsaco,'fused_rmsnorm_gate_up_silu_kernel')

args=[ctypes.c_void_p(xb.ptr),ctypes.c_void_p(wfnb.ptr),ctypes.c_void_p(wgb.ptr),ctypes.c_void_p(wub.ptr),
      ctypes.c_void_p(ob.ptr),ctypes.c_int(H),ctypes.c_int(FFN),ctypes.c_float(eps)]
block_size=256
grid=((FFN+block_size-1)//block_size,1,1)
hip.launch_kernel(function=fn,grid=grid,block=(block_size,1,1),args=args,shared_mem=0,expected_args=8)
hip.synchronize()
buf=bytearray(FFN*2)
hip.safe_memcpy_device_to_host(buf,ctypes.c_void_p(ob.ptr),tag='output_debug')
gpu=np.frombuffer(bytes(buf),dtype=np.float16).astype(np.float32)

def rms(xx,ww,eps):
    ms=np.mean(xx.astype(np.float64)**2); return (xx/np.sqrt(ms+eps)*ww).astype(np.float32)
xn = rms(x.astype(np.float32), w_fn.astype(np.float32), eps)
gate = xn@Wg.astype(np.float32).T
up_ = xn@Wu.astype(np.float32).T
silu = gate/(1.0+np.exp(-gate))
ref = silu*up_

print('diff max:', np.max(np.abs(gpu-ref)))
print('gpu[:6]=',gpu[:6])
print('ref[:6]=',ref[:6])
print('gpu[-6:]=',gpu[-6:])
print('ref[-6:]=',ref[-6:])
hip.cleanup()
