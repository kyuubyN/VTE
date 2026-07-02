import numpy as np, ctypes
from vte.core.model import VTEModel
from vte.bridge.memory import MemoryRegion

model = VTEModel.from_pretrained('qwen2.5:1.5b-q4_k_m', use_hip_graph=False)
print('=== LOADED OK ===')

def resolve_ptr(name):
    ptr = model.tensor_mapping.get(name)
    return ptr.ptr if hasattr(ptr, 'ptr') else ptr

def read_tensor(name, n_elements, dtype=np.float16):
    ptr_val = resolve_ptr(name)
    buf = bytearray(n_elements * np.dtype(dtype).itemsize)
    model._hip.safe_memcpy_device_to_host(buf, ctypes.c_void_p(ptr_val), tag='output_debug')
    return np.frombuffer(bytes(buf), dtype=dtype).astype(np.float32)

model.executor.prefill([46])
emb = read_tensor('input_embeddings', 1536)
norm_w = read_tensor('blk.0.attn_norm.weight', 1536)
gpu_out_full_model = read_tensor('blk.0.attn_norm.output', 1536)

# Agora roda o MESMO kernel rmsnorm, com os MESMOS dados, mas em buffers NOVOS e isolados
hip = model._hip
allocator = model._allocator
from vte.compiler.codegen import CodegenEngine

x_block = allocator.allocate(1536*2, 'iso_x', MemoryRegion.SCRATCH)
w_block = allocator.allocate(1536*2, 'iso_w', MemoryRegion.SCRATCH)
o_block = allocator.allocate(1536*2, 'iso_o', MemoryRegion.SCRATCH)

hip.safe_memcpy_host_to_device(ctypes.c_void_p(x_block.ptr), emb.astype(np.float16).tobytes(), tag='iso_x')
hip.safe_memcpy_host_to_device(ctypes.c_void_p(w_block.ptr), norm_w.astype(np.float16).tobytes(), tag='iso_w')

engine = CodegenEngine()
hsaco = engine.compile_kernel('rmsnorm', arch=hip.get_gpu_architecture(), tile_size=256)
mod, fn = hip.load_kernel(hsaco, 'rmsnorm_kernel')

args = [ctypes.c_void_p(x_block.ptr), ctypes.c_void_p(w_block.ptr), ctypes.c_void_p(o_block.ptr),
        ctypes.c_int(1536), ctypes.c_float(1e-5)]
hip.launch_kernel(function=fn, grid=(1,1,1), block=(256,1,1), args=args, shared_mem=0, expected_args=5)
hip.synchronize()

out_buf = bytearray(1536*2)
hip.safe_memcpy_device_to_host(out_buf, ctypes.c_void_p(o_block.ptr), tag='output_debug')
gpu_out_isolated = np.frombuffer(bytes(out_buf), dtype=np.float16).astype(np.float32)

ms = np.mean(emb.astype(np.float64)**2)
rms_inv = 1.0/np.sqrt(ms+1e-5)
ref = (emb.astype(np.float64)*rms_inv*norm_w.astype(np.float64)).astype(np.float32)

print('gpu_out (modelo completo)[:10]:', gpu_out_full_model[:10])
print('gpu_out (isolado, mesmos dados)[:10]:', gpu_out_isolated[:10])
print('ref (numpy)[:10]:', ref[:10])
print()
print('diff isolado vs ref:', np.max(np.abs(gpu_out_isolated - ref)))
print('diff isolado vs modelo completo:', np.max(np.abs(gpu_out_isolated - gpu_out_full_model)))

model.unload()
print('=== UNLOADED OK ===')
