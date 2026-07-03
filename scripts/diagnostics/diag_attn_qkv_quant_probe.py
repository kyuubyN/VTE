# Sonda de medicao isolada (plano: 41 -> 100 tok/s em batch=1, passo 2 --
# portar Q/K/V para dequant in-kernel). NAO muda nenhum caminho de producao:
# le os pesos REAIS de atencao (blk.0.attn_q/k/v/output.weight) direto do
# GGUF, ja crus em Q4_K/Q6_K, e mede se rotea-los para os kernels
# gemv_q4k/gemv_q6k (ja usados hoje so pelo FFN) seria correto e mais rapido
# do que o caminho atual (dequant no load -> gemv_coalesced em FP16).
#
# Duas licoes do incidente de TDR de 2026-07-02 aplicadas aqui desde o
# inicio (ver README, secao de bugs, "kernel-argument-count mismatch"):
#   1. gemv_coalesced/gemv_q4k/gemv_q6k tem 9 parametros (o ultimo e
#      residual_ptr, da Epilogue Fusion) -- NUNCA chamar com 8.
#   2. Sincronizar apos CADA lancamento individual, nunca enfileirar varios
#      lancamentos entre dois eventos sem synchronize() no meio -- o
#      limitador de duty cycle e o KernelWatchdog so recebem dados reais
#      dentro de synchronize().
import ctypes
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pathlib import Path
from vte.bridge.hip_runtime import HIPRuntime
from vte.bridge.memory import SlabAllocator, MemoryRegion
from vte.compiler.codegen import CodegenEngine
from vte.compiler.sanitizer import GGUFSanitizer
from vte.compiler.gguf_parser import GGUFParser
from vte.compiler.dequantizer import dequantize_q4_k, dequantize_q6_k, to_fp16_bytes

MODEL_PATH = Path("Model/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14

PROJECTIONS = [
    ("blk.0.attn_q.weight", 1536, 1536),
    ("blk.0.attn_k.weight", 256, 1536),
    ("blk.0.attn_v.weight", 256, 1536),
    ("blk.0.attn_output.weight", 1536, 1536),
]

N_TIMING_ITERS = 20
N_WARMUP_ITERS = 5


def load_raw_tensor_bytes(mmap_obj, tensor_info) -> bytes:
    offset = tensor_info["offset"]
    size = tensor_info["size"]
    return mmap_obj[offset:offset + size]


def main():
    print("=== Sonda C.1: quantizacao in-kernel para Q/K/V/O (plano 41->100 tps) ===\n")

    sanitizer = GGUFSanitizer(MODEL_PATH)
    sanitizer.validate()
    parser = GGUFParser(MODEL_PATH)
    tensors = parser.parse_tensors(sanitizer.header)

    import mmap
    f = open(MODEL_PATH, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    hip = HIPRuntime()
    hip.initialize()
    allocator = SlabAllocator(hip, 512 * 1024 * 1024)
    allocator.initialize()
    engine = CodegenEngine()
    arch = hip.get_gpu_architecture()

    hsaco_q4k = engine.compile_kernel("gemv_q4k", arch=arch)
    _, fn_q4k = hip.load_kernel(hsaco_q4k, "gemv_q4k_kernel")
    hsaco_q6k = engine.compile_kernel("gemv_q6k", arch=arch)
    _, fn_q6k = hip.load_kernel(hsaco_q6k, "gemv_q6k_kernel")
    hsaco_coal = engine.compile_kernel("gemv_coalesced", arch=arch)
    _, fn_coal = hip.load_kernel(hsaco_coal, "gemv_coalesced_kernel")

    def upload(arr, tag):
        block = allocator.allocate(len(arr.tobytes()), tag, MemoryRegion.SCRATCH)
        hip.safe_memcpy_host_to_device(ctypes.c_void_p(block.ptr), arr.tobytes(), tag=tag)
        return block

    def time_kernel(fn, args, grid, block, iters):
        # Sincroniza apos CADA lancamento -- nunca enfileirar sem dar ao
        # limitador de duty cycle/watchdog uma chance real de agir.
        for _ in range(N_WARMUP_ITERS):
            hip.launch_kernel(function=fn, args=args, grid=grid, block=block,
                               shared_mem=0, expected_args=9)
            hip.synchronize()

        total_ms = 0.0
        for _ in range(iters):
            start = hip.event_create()
            stop = hip.event_create()
            hip.event_record(start)
            hip.launch_kernel(function=fn, args=args, grid=grid, block=block,
                               shared_mem=0, expected_args=9)
            hip.event_record(stop)
            total_ms += hip.event_elapsed_ms(start, stop)
            hip.event_destroy(start)
            hip.event_destroy(stop)
            hip.synchronize()
        return total_ms / iters

    results = []

    for name, N, K in PROJECTIONS:
        tensor_info = tensors[name]
        dtype = tensor_info["dtype"]
        raw_bytes = load_raw_tensor_bytes(mm, tensor_info)

        print(f"--- {name}  shape=({N},{K})  dtype_gguf={'Q4_K' if dtype == GGML_TYPE_Q4_K else 'Q6_K'} ---")

        if dtype == GGML_TYPE_Q4_K:
            ref_w = dequantize_q4_k(raw_bytes, N * K).reshape(N, K)
            quant_fn, quant_name = fn_q4k, "gemv_q4k"
        elif dtype == GGML_TYPE_Q6_K:
            ref_w = dequantize_q6_k(raw_bytes, N * K).reshape(N, K)
            quant_fn, quant_name = fn_q6k, "gemv_q6k"
        else:
            print(f"  dtype inesperado ({dtype}), pulando.")
            continue

        fp16_w = np.frombuffer(to_fp16_bytes(raw_bytes, dtype, N * K), dtype=np.float16)

        for batch in (1, 4):
            rng = np.random.default_rng(hash(name) % (2**31))
            x = (rng.standard_normal((batch, K)) * 0.3).astype(np.float16)
            ref_out = x.astype(np.float32) @ ref_w.T

            xb = upload(x.reshape(-1), f"x_{name}_{batch}")
            wb_raw = upload(np.frombuffer(raw_bytes, dtype=np.uint8), f"wraw_{name}_{batch}")
            wb_fp16 = upload(fp16_w, f"wfp16_{name}_{batch}")
            ob_quant = allocator.allocate(N * batch * 2, f"o_quant_{name}_{batch}", MemoryRegion.SCRATCH)
            ob_coal = allocator.allocate(N * batch * 2, f"o_coal_{name}_{batch}", MemoryRegion.SCRATCH)

            # 9 argumentos (input, weight, output, batch, seq_len, in_features,
            # out_features, bias_ptr, residual_ptr) -- ver nota no topo.
            args_quant = [ctypes.c_void_p(xb.ptr), ctypes.c_void_p(wb_raw.ptr), ctypes.c_void_p(ob_quant.ptr),
                          ctypes.c_int(batch), ctypes.c_int(1), ctypes.c_int(K), ctypes.c_int(N),
                          ctypes.c_void_p(0), ctypes.c_void_p(0)]
            args_coal = [ctypes.c_void_p(xb.ptr), ctypes.c_void_p(wb_fp16.ptr), ctypes.c_void_p(ob_coal.ptr),
                         ctypes.c_int(batch), ctypes.c_int(1), ctypes.c_int(K), ctypes.c_int(N),
                         ctypes.c_void_p(0), ctypes.c_void_p(0)]

            grid = (N, batch, 1)
            block = (64, 1, 1)

            hip.launch_kernel(function=quant_fn, args=args_quant, grid=grid, block=block,
                               shared_mem=0, expected_args=9)
            hip.synchronize()
            hip.launch_kernel(function=fn_coal, args=args_coal, grid=grid, block=block,
                               shared_mem=0, expected_args=9)
            hip.synchronize()

            buf_quant = bytearray(N * batch * 2)
            hip.safe_memcpy_device_to_host(buf_quant, ctypes.c_void_p(ob_quant.ptr), tag="output_debug")
            gpu_quant = np.frombuffer(bytes(buf_quant), dtype=np.float16).astype(np.float32).reshape(batch, N)

            buf_coal = bytearray(N * batch * 2)
            hip.safe_memcpy_device_to_host(buf_coal, ctypes.c_void_p(ob_coal.ptr), tag="output_debug")
            gpu_coal = np.frombuffer(bytes(buf_coal), dtype=np.float16).astype(np.float32).reshape(batch, N)

            rel_vs_ref = np.max(np.abs(gpu_quant - ref_out) / (np.abs(ref_out) + 1e-3))
            rel_vs_prod = np.max(np.abs(gpu_quant - gpu_coal) / (np.abs(gpu_coal) + 1e-3))

            ms_quant = time_kernel(quant_fn, args_quant, grid, block, N_TIMING_ITERS)
            ms_coal = time_kernel(fn_coal, args_coal, grid, block, N_TIMING_ITERS)

            speedup = ms_coal / ms_quant if ms_quant > 0 else float("nan")

            print(f"  batch={batch}: rel_vs_numpy={rel_vs_ref:.5f}  rel_vs_producao_fp16={rel_vs_prod:.5f}  "
                  f"{quant_name}={ms_quant*1000:.2f}us  gemv_coalesced(fp16)={ms_coal*1000:.2f}us  "
                  f"speedup={speedup:.3f}x")

            results.append({
                "tensor": name, "batch": batch, "rel_vs_numpy": rel_vs_ref,
                "rel_vs_producao_fp16": rel_vs_prod, "us_quant": ms_quant * 1000,
                "us_fp16": ms_coal * 1000, "speedup": speedup,
            })
        print()

    print("=== Resumo ===")
    worst_rel = max(r["rel_vs_numpy"] for r in results)
    print(f"Pior erro relativo vs. referencia NumPy: {worst_rel:.5f} "
          f"({'OK (tolerancia 1e-2)' if worst_rel < 1e-2 else 'FALHOU tolerancia 1e-2'})")

    for batch in (1, 4):
        rows = [r for r in results if r["batch"] == batch]
        avg_speedup = sum(r["speedup"] for r in rows) / len(rows)
        total_us_quant = sum(r["us_quant"] for r in rows)
        total_us_fp16 = sum(r["us_fp16"] for r in rows)
        print(f"batch={batch}: speedup medio (kernel isolado) = {avg_speedup:.3f}x  "
              f"| soma Q/K/V/O: {total_us_fp16:.2f}us (fp16) -> {total_us_quant:.2f}us (in-kernel quant)  "
              f"({(total_us_fp16 - total_us_quant):.2f}us de diferenca por camada, x28 camadas = "
              f"{(total_us_fp16 - total_us_quant) * 28 / 1000:.3f}ms por token)")

    print("\nNota: isto mede SO o custo do kernel GEMV isolado (leitura de peso + "
          "compute), nao o pipeline completo. O ganho real em tok/s de producao "
          "so pode ser confirmado migrando de fato o roteamento e medindo "
          "generate() ponta a ponta.")

    hip.cleanup()
    mm.close()
    f.close()


if __name__ == "__main__":
    main()
