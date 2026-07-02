"""
Etapa I.2.a — Teste de canário do layout de batch no KV cache.

Antes de tocar em qualquer kernel (flash_attention/rope), valida que a
matemática de stride (`layer_kv_size_per_batch`, `kv_batch_stride_elements`)
não faz sequências do batch se sobreporem. Escreve um valor identificável
(float(batch_idx)) em cada slot via memcpy direto (sem kernel nenhum) e lê
de volta, verificando que cada sequência só contém o próprio valor.
"""
import numpy as np
import ctypes
from vte.bridge.hip_runtime import HIPRuntime
from vte.bridge.memory import SlabAllocator, MemoryRegion
from vte.compiler.qwen_mapper import QwenTensorMapper


class FakeParser:
    tensors = {}  # sem pesos: só queremos testar o layout de KV cache/arena


BATCH_SIZE = 4
CONTEXT_LENGTH = 64  # pequeno de propósito, só para o teste ser rápido
LAYERS = 28
KV_HEADS = 2
HEAD_DIM = 128

metadata = {
    "block_count": LAYERS,
    "attention.head_count_kv": KV_HEADS,
    "attention.key_length": HEAD_DIM,
    "embedding_length": 1536,
}

hip = HIPRuntime(); hip.initialize()
allocator = SlabAllocator(hip, 512 * 1024 * 1024); allocator.initialize()

mapper = QwenTensorMapper(FakeParser(), metadata)
tensor_mapping = mapper.map_and_allocate_tensors(
    allocator, hip, profiler=None, context_length=CONTEXT_LENGTH, batch_size=BATCH_SIZE
)

kv_stride_elements = tensor_mapping['kv_batch_stride_elements']
kv_stride_bytes = kv_stride_elements * 2  # FP16
print(f"kv_batch_stride_elements = {kv_stride_elements} "
      f"(esperado: {KV_HEADS * HEAD_DIM * CONTEXT_LENGTH})")
assert kv_stride_elements == KV_HEADS * HEAD_DIM * CONTEXT_LENGTH, "stride de batch incorreto!"

print("\n=== Escrevendo padrão identificável por (layer, K/V, batch_idx) ===")
n_elements_per_slot = KV_HEADS * HEAD_DIM * CONTEXT_LENGTH

errors = 0
checked_layers = [0, 1, LAYERS - 1]  # amostragem: primeira, segunda e última camada
for l in checked_layers:
    k_base = tensor_mapping[f'blk.{l}.kv_cache.k']
    v_base = tensor_mapping[f'blk.{l}.kv_cache.v']
    k_base_ptr = k_base.ptr if hasattr(k_base, 'ptr') else k_base
    v_base_ptr = v_base.ptr if hasattr(v_base, 'ptr') else v_base

    # Escreve: slot de K da sequência b preenchido com valor (100 + l*10 + b);
    # slot de V da sequência b preenchido com valor -(100 + l*10 + b) (sinal
    # oposto para detectar cruzamento K<->V também, não só entre sequências).
    for b in range(BATCH_SIZE):
        val_k = 100.0 + l * 10 + b
        val_v = -(100.0 + l * 10 + b)
        arr_k = np.full(n_elements_per_slot, val_k, dtype=np.float16)
        arr_v = np.full(n_elements_per_slot, val_v, dtype=np.float16)
        offset_bytes = b * kv_stride_bytes
        hip.safe_memcpy_host_to_device(
            ctypes.c_void_p(k_base_ptr + offset_bytes), arr_k.tobytes(), tag=f"canary_k_l{l}_b{b}"
        )
        hip.safe_memcpy_host_to_device(
            ctypes.c_void_p(v_base_ptr + offset_bytes), arr_v.tobytes(), tag=f"canary_v_l{l}_b{b}"
        )

    hip.synchronize()

    # Lê de volta e verifica que cada slot só contém o valor esperado (sem
    # vazamento de sequências vizinhas nem entre K e V).
    for b in range(BATCH_SIZE):
        expected_k = 100.0 + l * 10 + b
        expected_v = -(100.0 + l * 10 + b)
        offset_bytes = b * kv_stride_bytes

        buf_k = bytearray(n_elements_per_slot * 2)
        hip.safe_memcpy_device_to_host(buf_k, ctypes.c_void_p(k_base_ptr + offset_bytes), tag="output")
        got_k = np.frombuffer(bytes(buf_k), dtype=np.float16)

        buf_v = bytearray(n_elements_per_slot * 2)
        hip.safe_memcpy_device_to_host(buf_v, ctypes.c_void_p(v_base_ptr + offset_bytes), tag="output")
        got_v = np.frombuffer(bytes(buf_v), dtype=np.float16)

        ok_k = np.all(got_k == np.float16(expected_k))
        ok_v = np.all(got_v == np.float16(expected_v))
        status_k = "OK" if ok_k else f"FALHOU (esperado {expected_k}, achou min={got_k.min()} max={got_k.max()})"
        status_v = "OK" if ok_v else f"FALHOU (esperado {expected_v}, achou min={got_v.min()} max={got_v.max()})"
        print(f"  layer={l:<3} batch={b}: K={status_k}  V={status_v}")
        if not ok_k or not ok_v:
            errors += 1

print(f"\n{'PASS — nenhum vazamento entre sequencias do batch nem entre K/V' if errors == 0 else f'FALHOU: {errors} slots corrompidos'}")
hip.cleanup()
