"""
Etapa I.1 (Fase I — Batched Decode): prova de hipótese isolada.

Pergunta: para gemv_coalesced (já batch-capaz via m=blockIdx.y=batch*seq_len),
o throughput agregado escala ~N x ao processar N linhas simultâneas, com o
mesmo peso lido 1x da VRAM? Ou fica achatado como as otimizações anteriores
de batch=1 (pre-pack de endereçamento, remoção de ALU do unpack) que
mediram flat/negativo?

Critério de decisão (definido no plano): speedup agregado >= 0.7 x N até
N=8 para justificar a Etapa I.2. Sem essa evidência, a Fase I inteira para
aqui — sem tocar em nenhum arquivo de produção.

Zero mudança de kernel/produção: só invoca gemv_coalesced com batch real
em vez do hardcoded 1.
"""
import numpy as np
import ctypes
from vte.bridge.hip_runtime import HIPRuntime
from vte.bridge.memory import SlabAllocator, MemoryRegion
from vte.compiler.codegen import CodegenEngine

hip = HIPRuntime(); hip.initialize()
allocator = SlabAllocator(hip, 1024 * 1024 * 1024); allocator.initialize()

def up(arr, tag):
    b = allocator.allocate(len(arr.tobytes()), tag, MemoryRegion.SCRATCH)
    hip.safe_memcpy_host_to_device(ctypes.c_void_p(b.ptr), arr.tobytes(), tag=tag)
    return b

eng = CodegenEngine(); arch = hip.get_gpu_architecture()
hsaco = eng.compile_kernel('gemv_coalesced', arch=arch)
mod, fn = hip.load_kernel(hsaco, 'gemv_coalesced_kernel')

BLOCK = 64
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
SHAPES = [
    ("gate/up",       1536, 8960),
    ("down_proj",     8960, 1536),
    ("attn_output",   1536, 1536),
    ("q_proj",        1536, 1536),
]


def numeric_check(in_features, out_features, batch, seed):
    """Valida LINHA POR LINHA do batch contra X @ W.T em NumPy. Retorna pior diff."""
    rng = np.random.default_rng(seed)
    # Linhas DISTINTAS (crítico: linhas idênticas mascarariam bug de indexação por m).
    X = (rng.standard_normal((batch, in_features)) * 0.3).astype(np.float16)
    W = (rng.standard_normal((out_features, in_features)) * 0.05).astype(np.float16)

    xb = up(X.reshape(-1), f'x_nc_{seed}')
    wb = up(W.reshape(-1), f'w_nc_{seed}')
    ob = allocator.allocate(batch * out_features * 2, f'o_nc_{seed}', MemoryRegion.SCRATCH)

    args = [ctypes.c_void_p(xb.ptr), ctypes.c_void_p(wb.ptr), ctypes.c_void_p(ob.ptr),
            ctypes.c_int(batch), ctypes.c_int(1), ctypes.c_int(in_features), ctypes.c_int(out_features),
            ctypes.c_void_p(0), ctypes.c_void_p(0)]
    grid = (out_features, batch, 1)
    hip.launch_kernel(function=fn, grid=grid, block=(BLOCK, 1, 1), args=args, shared_mem=0, expected_args=9)
    hip.synchronize()

    buf = bytearray(batch * out_features * 2)
    hip.safe_memcpy_device_to_host(buf, ctypes.c_void_p(ob.ptr), tag='output')
    gpu = np.frombuffer(bytes(buf), dtype=np.float16).astype(np.float32).reshape(batch, out_features)
    ref = X.astype(np.float32) @ W.astype(np.float32).T

    worst_row_diff = 0.0
    worst_row = -1
    for b in range(batch):
        d = np.max(np.abs(gpu[b] - ref[b]))
        if d > worst_row_diff:
            worst_row_diff = d
            worst_row = b
    return worst_row_diff, worst_row


def bench(in_features, out_features, batch, iters=50, warmup=20):
    rng = np.random.default_rng(123)
    X = (rng.standard_normal((batch, in_features)) * 0.3).astype(np.float16)
    W = (rng.standard_normal((out_features, in_features)) * 0.05).astype(np.float16)
    xb = up(X.reshape(-1), f'x_b{batch}_{in_features}_{out_features}')
    wb = up(W.reshape(-1), f'w_b{batch}_{in_features}_{out_features}')
    ob = allocator.allocate(batch * out_features * 2, f'o_b{batch}_{in_features}_{out_features}', MemoryRegion.SCRATCH)

    args = [ctypes.c_void_p(xb.ptr), ctypes.c_void_p(wb.ptr), ctypes.c_void_p(ob.ptr),
            ctypes.c_int(batch), ctypes.c_int(1), ctypes.c_int(in_features), ctypes.c_int(out_features),
            ctypes.c_void_p(0), ctypes.c_void_p(0)]
    grid = (out_features, batch, 1)
    block = (BLOCK, 1, 1)

    for _ in range(warmup):
        hip.launch_kernel(function=fn, grid=grid, block=block, args=args, shared_mem=0, expected_args=9)
    hip.synchronize()

    times_us = []
    for _ in range(iters):
        ev_start = hip.event_create(); ev_stop = hip.event_create()
        hip.event_record(ev_start)
        hip.launch_kernel(function=fn, grid=grid, block=block, args=args, shared_mem=0, expected_args=9)
        hip.event_record(ev_stop)
        ms = hip.event_elapsed_ms(ev_start, ev_stop)
        hip.event_destroy(ev_start); hip.event_destroy(ev_stop)
        times_us.append(ms * 1000.0)

    return float(np.median(times_us))


def main():
    print("=" * 78)
    print(" Etapa I.1 — Validação numérica (linha por linha do batch)")
    print("=" * 78)
    worst_overall = 0.0
    for name, in_f, out_f in SHAPES:
        for batch in [1, 4, 8]:
            diff, row = numeric_check(in_f, out_f, batch, seed=hash((name, batch)) % (2**31))
            status = "PASS" if diff < 2e-3 else "FAIL"
            print(f"  {name:<14} in={in_f:<5} out={out_f:<5} batch={batch:<3} "
                  f"pior_diff={diff:.6f} (linha {row})  [{status}]")
            worst_overall = max(worst_overall, diff)
    print(f"\nPior diff geral: {worst_overall:.6f}  "
          f"({'TODOS PASS @ 2e-3' if worst_overall < 2e-3 else 'ALGUMA LINHA FALHOU'})")

    print()
    print("=" * 78)
    print(" Etapa I.1 — Escalonamento de throughput agregado por batch_size")
    print("=" * 78)

    results = {}
    for name, in_f, out_f in SHAPES:
        print(f"\n--- {name} (in={in_f}, out={out_f}) ---")
        t1 = None
        for batch in BATCH_SIZES:
            us = bench(in_f, out_f, batch)
            if t1 is None:
                t1 = us
            ideal_us = t1  # tempo ideal seria igual ao de batch=1 (peso lido 1x, N linhas de graça)
            speedup_agregado = (batch * t1) / us  # "tok/s agregado" normalizado
            eficiencia = speedup_agregado / batch
            print(f"  batch={batch:<3} {us:8.2f} us/launch   "
                  f"speedup_agregado={speedup_agregado:5.2f}x   eficiencia={eficiencia*100:5.1f}%")
            results.setdefault(name, []).append((batch, us, speedup_agregado, eficiencia))

    print()
    print("=" * 78)
    print(" VEREDITO (critério: speedup_agregado >= 0.7 x batch até batch=8)")
    print("=" * 78)
    all_pass = True
    for name, rows in results.items():
        for batch, us, speedup, eff in rows:
            if batch <= 8:
                ok = speedup >= 0.7 * batch
                if not ok:
                    all_pass = False
                print(f"  {name:<14} batch={batch:<3} speedup={speedup:5.2f}x "
                      f"(precisa >= {0.7*batch:4.2f}x)  [{'OK' if ok else 'FALHOU'}]")
    print()
    if all_pass:
        print(">>> HIPOTESE CONFIRMADA: prosseguir para Etapa I.2 (batch estatico em lockstep).")
    else:
        print(">>> HIPOTESE REFUTADA (ou parcialmente): NAO prosseguir para I.2 sem reavaliar.")

    hip.cleanup()


if __name__ == "__main__":
    main()
