import numpy as np, ctypes
from vte.bridge.hip_runtime import HIPRuntime
from vte.bridge.memory import SlabAllocator, MemoryRegion
from vte.compiler.codegen import CodegenEngine
from vte.compiler.rope_cache_builder import RoPECacheBuilder

hip = HIPRuntime()
hip.initialize()
allocator = SlabAllocator(hip, 16*1024*1024)
allocator.initialize()

head_dim = 128
num_q_heads = 2
num_kv_heads = 1
max_seq_len = 32
theta = 10000.0

builder = RoPECacheBuilder(max_seq_len=max_seq_len, head_dim=head_dim, rope_theta=theta)
cos_cache, sin_cache = builder.build_cache()
cos_ptr, sin_ptr = builder.upload_to_vram(cos_cache, sin_cache, hip, allocator)

np.random.seed(3)
seq_len = 1
position = 5  # posicao NAO-zero, onde o bug de indexacao se manifestaria
q = (np.random.randn(seq_len*num_q_heads*head_dim)*0.5).astype(np.float16)
k = (np.random.randn(seq_len*num_kv_heads*head_dim)*0.5).astype(np.float16)

q_block = allocator.allocate(len(q)*2, 'q', MemoryRegion.SCRATCH)
k_block = allocator.allocate(len(k)*2, 'k', MemoryRegion.SCRATCH)
offset_block = allocator.allocate(4, 'off', MemoryRegion.SCRATCH)

hip.safe_memcpy_host_to_device(ctypes.c_void_p(q_block.ptr), q.tobytes(), tag='q')
hip.safe_memcpy_host_to_device(ctypes.c_void_p(k_block.ptr), k.tobytes(), tag='k')
hip.safe_memcpy_host_to_device(ctypes.c_void_p(offset_block.ptr), np.array([position], dtype=np.int32).tobytes(), tag='off')

engine = CodegenEngine()
hsaco = engine.compile_kernel('rope', arch=hip.get_gpu_architecture())
mod, fn = hip.load_kernel(hsaco, 'rope_kernel')

args = [ctypes.c_void_p(q_block.ptr), ctypes.c_void_p(k_block.ptr), ctypes.c_void_p(cos_ptr), ctypes.c_void_p(sin_ptr),
        ctypes.c_int(seq_len), ctypes.c_int(num_q_heads), ctypes.c_int(num_kv_heads), ctypes.c_int(head_dim),
        ctypes.c_void_p(offset_block.ptr)]

total = max(seq_len*num_q_heads*head_dim, seq_len*num_kv_heads*head_dim)
grid = ((total+255)//256, 1, 1)
hip.launch_kernel(function=fn, grid=grid, block=(256,1,1), args=args, shared_mem=0, expected_args=9)
hip.synchronize()

q_out_buf = bytearray(len(q)*2)
hip.safe_memcpy_device_to_host(q_out_buf, ctypes.c_void_p(q_block.ptr), tag='output_debug')
q_out = np.frombuffer(bytes(q_out_buf), dtype=np.float16).astype(np.float32)

# Referencia numpy (rotate-half padrao)
def rope_ref(x, pos, head_dim, theta):
    half = head_dim // 2
    freqs = 1.0/(theta**(np.arange(0,half)/half))
    angles = pos*freqs
    cos_v = np.cos(angles); sin_v = np.sin(angles)
    x = x.reshape(-1, head_dim).astype(np.float64)
    out = np.zeros_like(x)
    for h in range(x.shape[0]):
        x1 = x[h,:half]; x2 = x[h,half:]
        out[h,:half] = x1*cos_v - x2*sin_v
        out[h,half:] = x1*sin_v + x2*cos_v
    return out.reshape(-1).astype(np.float32)

ref = rope_ref(q, position, head_dim, theta)
print('GPU q_out[:10]:', q_out[:10])
print('REF q_out[:10]:', ref[:10])
print('diff max:', np.max(np.abs(q_out-ref)))

hip.cleanup()
