import numpy as np, ctypes
from vte.bridge.hip_runtime import HIPRuntime
from vte.bridge.memory import SlabAllocator, MemoryRegion
from vte.compiler.codegen import CodegenEngine
from vte.compiler.dequantizer import dequantize_q6_k

hip = HIPRuntime(); hip.initialize()
allocator = SlabAllocator(hip, 256*1024*1024); allocator.initialize()

def up(arr, tag):
    b = allocator.allocate(len(arr.tobytes()), tag, MemoryRegion.SCRATCH)
    hip.safe_memcpy_host_to_device(ctypes.c_void_p(b.ptr), arr.tobytes(), tag=tag); return b

def make_q6k(n_blocks, seed):
    rng = np.random.default_rng(seed)
    blocks = np.zeros((n_blocks, 210), dtype=np.uint8)
    blocks[:, 0:208] = rng.integers(0, 256, size=(n_blocks, 208), dtype=np.uint8)  # ql+qh+scales
    d = (rng.random(n_blocks).astype(np.float32) * 0.02 + 0.01).astype(np.float16)
    blocks[:, 208:210] = d.view(np.uint8).reshape(n_blocks, 2)
    return blocks.reshape(-1).tobytes()

eng = CodegenEngine(); arch = hip.get_gpu_architecture()
hsaco = eng.compile_kernel('gemv_q6k', arch=arch)
mod, fn = hip.load_kernel(hsaco, 'gemv_q6k_kernel')

def run_case(N, K, seed):
    n_blocks = N * (K // 256)
    raw = make_q6k(n_blocks, seed)
    Wref = dequantize_q6_k(raw, N * K).reshape(N, K)
    rng = np.random.default_rng(seed + 100)
    x = (rng.standard_normal(K) * 0.3).astype(np.float16)
    xb = up(x, f'x{seed}'); wb = up(np.frombuffer(raw, dtype=np.uint8), f'w{seed}')
    ob = allocator.allocate(N * 2, f'o{seed}', MemoryRegion.SCRATCH)
    args = [ctypes.c_void_p(xb.ptr), ctypes.c_void_p(wb.ptr), ctypes.c_void_p(ob.ptr),
            ctypes.c_int(1), ctypes.c_int(1), ctypes.c_int(K), ctypes.c_int(N), ctypes.c_void_p(0)]
    hip.launch_kernel(function=fn, grid=(N,1,1), block=(64,1,1), args=args, shared_mem=0, expected_args=8)
    hip.synchronize()
    buf = bytearray(N * 2)
    hip.safe_memcpy_device_to_host(buf, ctypes.c_void_p(ob.ptr), tag='output')
    gpu = np.frombuffer(bytes(buf), dtype=np.float16).astype(np.float32)
    ref = x.astype(np.float32) @ Wref.T
    rel = np.max(np.abs(gpu - ref) / (np.abs(ref) + 1e-3))
    print(f"  N={N} K={K}: rel={rel:.5f} abs={np.max(np.abs(gpu-ref)):.4f}  gpu[:3]={gpu[:3]} ref[:3]={ref[:3]}")
    return rel

print("=== Validacao Q6_K in-kernel vs dequantize_q6_k ===")
d = [run_case(1536, 8960, 1), run_case(256, 8960, 2), run_case(512, 256, 3)]
worst = max(d)
print(f"PIOR rel = {worst:.5f}  ({'PASS' if worst < 1e-2 else 'FAIL'} @ rtol 1e-2)")
hip.cleanup()
