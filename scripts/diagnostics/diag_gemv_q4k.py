"""Validação do gemv_q4k_kernel (Etapa C): desquant Q4_K in-kernel + GEMV
   contra a referência Python já validada (dequantize_q4_k ~ gguf)."""
import numpy as np, ctypes, subprocess, os
from vte.bridge.hip_runtime import HIPRuntime
from vte.bridge.memory import SlabAllocator, MemoryRegion
from vte.compiler.codegen import CodegenEngine
from vte.compiler.dequantizer import dequantize_q4_k

hip = HIPRuntime(); hip.initialize()
allocator = SlabAllocator(hip, 256*1024*1024); allocator.initialize()

def up(arr, tag):
    b = allocator.allocate(len(arr.tobytes()), tag, MemoryRegion.SCRATCH)
    hip.safe_memcpy_host_to_device(ctypes.c_void_p(b.ptr), arr.tobytes(), tag=tag)
    return b

def make_q4k_blocks(n_blocks, seed):
    """Constrói bytes Q4_K crus com d/dmin sãos e scales/qs aleatórios."""
    rng = np.random.default_rng(seed)
    blocks = np.zeros((n_blocks, 144), dtype=np.uint8)
    d = (rng.random(n_blocks).astype(np.float32) * 0.03 + 0.02).astype(np.float16)
    dmin = (rng.random(n_blocks).astype(np.float32) * 0.05 + 0.03).astype(np.float16)
    blocks[:, 0:2] = d.view(np.uint8).reshape(n_blocks, 2)
    blocks[:, 2:4] = dmin.view(np.uint8).reshape(n_blocks, 2)
    blocks[:, 4:144] = rng.integers(0, 256, size=(n_blocks, 140), dtype=np.uint8)
    return blocks.reshape(-1).tobytes()

eng = CodegenEngine(); arch = hip.get_gpu_architecture()
hsaco = eng.compile_kernel('gemv_q4k', arch=arch)
mod, fn = hip.load_kernel(hsaco, 'gemv_q4k_kernel')

def run_case(N, K, seed):
    assert K % 256 == 0
    n_blocks = N * (K // 256)
    raw = make_q4k_blocks(n_blocks, seed)
    Wref = dequantize_q4_k(raw, N * K).reshape(N, K)   # referência FP32

    rng = np.random.default_rng(seed + 100)
    x = (rng.standard_normal(K) * 0.3).astype(np.float16)

    xb = up(x, f'x{seed}')
    wb = up(np.frombuffer(raw, dtype=np.uint8), f'w{seed}')
    ob = allocator.allocate(N * 2, f'o{seed}', MemoryRegion.SCRATCH)

    args = [ctypes.c_void_p(xb.ptr), ctypes.c_void_p(wb.ptr), ctypes.c_void_p(ob.ptr),
            ctypes.c_int(1), ctypes.c_int(1), ctypes.c_int(K), ctypes.c_int(N),
            ctypes.c_void_p(0)]
    hip.launch_kernel(function=fn, grid=(N, 1, 1), block=(64, 1, 1), args=args,
                      shared_mem=0, expected_args=8)
    hip.synchronize()

    buf = bytearray(N * 2)
    hip.safe_memcpy_device_to_host(buf, ctypes.c_void_p(ob.ptr), tag='output')
    gpu = np.frombuffer(bytes(buf), dtype=np.float16).astype(np.float32)
    ref = x.astype(np.float32) @ Wref.T

    diff = np.max(np.abs(gpu - ref))
    # Tolerância RELATIVA: saídas magnitude ~250 -> atol grande é FP16-noise.
    rel = np.max(np.abs(gpu - ref) / (np.abs(ref) + 1e-3))
    print(f"  N={N} K={K}: abs_diff={diff:.5f} rel={rel:.5f}  gpu[:3]={gpu[:3]} ref[:3]={ref[:3]}")
    return rel

print("=== Validação Q4_K in-kernel vs referência dequantize_q4_k ===")
d = [run_case(256, 8960, 1),    # down_proj-shape
     run_case(1536, 8960, 2),   # down_proj real (N=1536)
     run_case(512, 256, 3)]     # 1 super-bloco por linha
worst = max(d)
print(f"PIOR rel = {worst:.5f}  ({'PASS' if worst < 1e-2 else 'FAIL'} @ rtol 1e-2)")

print("\n=== ISA ===")
rb = r"C:\Program Files\AMD\ROCm\6.4\bin"
raw_o = os.path.join(os.environ.get("TEMP", "/tmp"), "gemv_q4k_gfx1102.o")
subprocess.run([os.path.join(rb, "clang-offload-bundler.exe"), "--type=o", "--unbundle",
                f"--input={hsaco}", "--targets=hipv4-amdgcn-amd-amdhsa--gfx1102",
                f"--output={raw_o}"], capture_output=True)
out = subprocess.run([os.path.join(rb, "llvm-objdump.exe"), "-d", raw_o],
                     capture_output=True, text=True).stdout
print(f"  v_bfe (bit-field extract) = {out.count('v_bfe')}  "
      f"ds_swizzle/bpermute = {out.count('ds_swizzle')+out.count('ds_bpermute')}")
hip.cleanup()
