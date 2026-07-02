"""Validação do gemv_coalesced_kernel (Etapa B / Split-K): numérica + ISA."""
import numpy as np, ctypes, subprocess, os
from vte.bridge.hip_runtime import HIPRuntime
from vte.bridge.memory import SlabAllocator, MemoryRegion
from vte.compiler.codegen import CodegenEngine

hip = HIPRuntime(); hip.initialize()
allocator = SlabAllocator(hip, 512*1024*1024); allocator.initialize()

def up(arr, tag):
    b = allocator.allocate(len(arr.tobytes()), tag, MemoryRegion.SCRATCH)
    hip.safe_memcpy_host_to_device(ctypes.c_void_p(b.ptr), arr.tobytes(), tag=tag)
    return b

eng = CodegenEngine(); arch = hip.get_gpu_architecture()
hsaco = eng.compile_kernel('gemv_coalesced', arch=arch)
mod, fn = hip.load_kernel(hsaco, 'gemv_coalesced_kernel')

BLOCK = 64

def run_case(in_features, out_features, has_bias, seed):
    np.random.seed(seed)
    x = (np.random.randn(in_features) * 0.3).astype(np.float16)
    W = (np.random.randn(out_features, in_features) * 0.05).astype(np.float16)
    bias = (np.random.randn(out_features) * 0.5).astype(np.float16) if has_bias else None
    xb = up(x, f'x{seed}'); wb = up(W.reshape(-1), f'w{seed}')
    ob = allocator.allocate(out_features * 2, f'o{seed}', MemoryRegion.SCRATCH)
    bb = up(bias, f'b{seed}') if has_bias else None
    args = [ctypes.c_void_p(xb.ptr), ctypes.c_void_p(wb.ptr), ctypes.c_void_p(ob.ptr),
            ctypes.c_int(1), ctypes.c_int(1), ctypes.c_int(in_features), ctypes.c_int(out_features),
            ctypes.c_void_p(bb.ptr if has_bias else 0)]
    grid = (out_features, 1, 1)
    hip.launch_kernel(function=fn, grid=grid, block=(BLOCK, 1, 1), args=args,
                      shared_mem=0, expected_args=8)
    hip.synchronize()
    buf = bytearray(out_features * 2)
    hip.safe_memcpy_device_to_host(buf, ctypes.c_void_p(ob.ptr), tag='output')
    gpu = np.frombuffer(bytes(buf), dtype=np.float16).astype(np.float32)
    ref = x.astype(np.float32) @ W.astype(np.float32).T
    if has_bias: ref = ref + bias.astype(np.float32)
    diff = np.max(np.abs(gpu - ref))
    print(f"  in={in_features} out={out_features} bias={has_bias}: diff_max={diff:.6f}  "
          f"gpu[:3]={gpu[:3]} ref[:3]={ref[:3]}")
    return diff

print("=== Validação numérica (BLOCK=64, 1 bloco/neurônio) ===")
d = [run_case(1536, 8960, False, 1),   # gate/up
     run_case(1536, 8960, False, 2),
     run_case(8960, 1536, False, 3),   # down (K=8960)
     run_case(1536, 1536, True, 4)]    # q_proj com bias
worst = max(d)
print(f"PIOR diff = {worst:.6f}  ({'PASS' if worst < 2e-3 else 'FAIL'} @ tol 2e-3)")

print("\n=== ISA (global_load_b128 + __shfl_down via ds_swizzle/ds_bpermute) ===")
rb = r"C:\Program Files\AMD\ROCm\6.4\bin"
raw = os.path.join(os.environ.get("TEMP", "/tmp"), "gemv_coalesced_gfx1102.o")
subprocess.run([os.path.join(rb, "clang-offload-bundler.exe"), "--type=o", "--unbundle",
                f"--input={hsaco}", "--targets=hipv4-amdgcn-amd-amdhsa--gfx1102",
                f"--output={raw}"], capture_output=True)
out = subprocess.run([os.path.join(rb, "llvm-objdump.exe"), "-d", raw],
                     capture_output=True, text=True).stdout
b128 = out.count("global_load_b128")
u16 = out.count("global_load_u16")
shfl = out.count("ds_swizzle") + out.count("ds_bpermute") + out.count("ds_permute")
print(f"  global_load_b128 = {b128}  (peso coalescido)")
print(f"  global_load_u16  = {u16}  (cauda, esperado ~0 p/ dims %8==0)")
print(f"  ds_swizzle/bpermute (shfl reduction) = {shfl}")
print(f"  VETORIZAÇÃO {'CONFIRMADA' if b128 > 0 else 'FALHOU'}")
hip.cleanup()
