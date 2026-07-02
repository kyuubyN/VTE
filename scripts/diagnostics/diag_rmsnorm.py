import numpy as np, ctypes
from vte.bridge.hip_runtime import HIPRuntime
from vte.bridge.memory import SlabAllocator, MemoryRegion
from vte.compiler.codegen import CodegenEngine

hip = HIPRuntime()
hip.initialize()
allocator = SlabAllocator(hip, 8*1024*1024)
allocator.initialize()

hidden_size = 1536
eps = 1e-5

def run_rmsnorm(x, w):
    x_block = allocator.allocate(hidden_size*2, 'x', MemoryRegion.SCRATCH)
    w_block = allocator.allocate(hidden_size*2, 'w', MemoryRegion.SCRATCH)
    o_block = allocator.allocate(hidden_size*2, 'o', MemoryRegion.SCRATCH)

    hip.safe_memcpy_host_to_device(ctypes.c_void_p(x_block.ptr), x.astype(np.float16).tobytes(), tag='x')
    hip.safe_memcpy_host_to_device(ctypes.c_void_p(w_block.ptr), w.astype(np.float16).tobytes(), tag='w')

    engine = CodegenEngine()
    hsaco = engine.compile_kernel('rmsnorm', arch=hip.get_gpu_architecture(), tile_size=256)
    mod, fn = hip.load_kernel(hsaco, 'rmsnorm_kernel')

    args = [ctypes.c_void_p(x_block.ptr), ctypes.c_void_p(w_block.ptr), ctypes.c_void_p(o_block.ptr),
            ctypes.c_int(hidden_size), ctypes.c_float(eps)]
    hip.launch_kernel(function=fn, grid=(1,1,1), block=(256,1,1), args=args, shared_mem=0, expected_args=5)
    hip.synchronize()

    out_buf = bytearray(hidden_size*2)
    hip.safe_memcpy_device_to_host(out_buf, ctypes.c_void_p(o_block.ptr), tag='output_debug')
    return np.frombuffer(bytes(out_buf), dtype=np.float16).astype(np.float32)

def rmsnorm_ref(x, w, eps=1e-5):
    ms = np.mean(x.astype(np.float64)**2)
    return (x / np.sqrt(ms + eps) * w).astype(np.float32)

np.random.seed(1)
w = np.random.randn(hidden_size).astype(np.float32) * 0.1 + 1.0

x1 = np.random.randn(hidden_size).astype(np.float32) * 0.02
x2 = np.random.randn(hidden_size).astype(np.float32) * 5.0  # bem diferente de x1

out1 = run_rmsnorm(x1, w)
out2 = run_rmsnorm(x2, w)
ref1 = rmsnorm_ref(x1, w)
ref2 = rmsnorm_ref(x2, w)

print('GPU out1 first5:', out1[:5])
print('REF out1 first5:', ref1[:5])
print('GPU out2 first5:', out2[:5])
print('REF out2 first5:', ref2[:5])
print('diff GPU out1 vs out2 (deveria ser grande, inputs bem diferentes):', np.max(np.abs(out1-out2)))
print('diff GPU vs REF (out1):', np.max(np.abs(out1-ref1)))
print('diff GPU vs REF (out2):', np.max(np.abs(out2-ref2)))

hip.cleanup()
