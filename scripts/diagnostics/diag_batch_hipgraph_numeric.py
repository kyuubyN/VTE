"""
Etapa I.2 (extensão HIP Graph) — Validação numérica do BatchedHIPGraphExecutor.

Compara BatchedHIPGraphExecutor(batch_size=K) contra BatchedFallbackExecutor
(eager, já validado bit-a-bit correto na Etapa I.2) rodando as MESMAS K
sequências — tensor_mappings SEPARADOS (pesos carregados 2x) para isolar
completamente os dois caminhos.
"""
import os
import glob
import numpy as np
import ctypes
from pathlib import Path

from vte.compiler.sanitizer import GGUFSanitizer
from vte.compiler.gguf_parser import GGUFParser
from vte.compiler.qwen_mapper import QwenTensorMapper, ActivationArena, is_raw_q4k_weight, is_raw_q6k_weight
from vte.compiler.qwen_compute import QwenComputeGraphBuilder
from vte.compiler.weight_loader import GGUFWeightLoader
from vte.bridge.hip_runtime import HIPRuntime
from vte.bridge.memory import SlabAllocator, MemoryRegion
from vte.core.fallback_executor import register_raw_q4k_weights, register_raw_q6k_weights
from vte.core.batched_fallback_executor import BatchedFallbackExecutor
from vte.core.batched_hip_graph_executor import BatchedHIPGraphExecutor

BATCH_SIZE = 4
SEQ_TOKENS = int(os.environ.get("VTE_DIAG_SEQ_TOKENS", "5"))
PROMPTS_TOKENS = [
    [791, 6864, 315, 9822, 374],
    [40, 1093, 311, 1855, 264],
    [791, 25342, 574, 12703, 304],
    [2323, 5652, 264, 3488, 922],
]
PROMPTS_TOKENS = [p[:SEQ_TOKENS] for p in PROMPTS_TOKENS]
assert len(PROMPTS_TOKENS) == BATCH_SIZE

model_path = Path(glob.glob("Model/*.gguf")[0])

hip = HIPRuntime(); hip.initialize()
vram_total = hip.get_device_properties()['total_global_mem']
allocator = SlabAllocator(hip, vram_total, requested_pool_size=int(vram_total * 0.7))
allocator.initialize()

sanitizer = GGUFSanitizer(model_path); sanitizer.validate()
parser = GGUFParser(model_path); parser.parse_tensors(sanitizer.header)

metadata = {
    "block_count": sanitizer.header.block_count,
    "context_length": 256,
    "embedding_length": sanitizer.header.embedding_length,
    "attention.head_count": 12,
    "attention.head_count_kv": 2,
    "attention.key_length": 128,
    "feed_forward_length": 8960,
    "rope.freq_base": 1000000.0,
    "attention.layer_norm_rms_epsilon": 1e-6,
}
CONTEXT_LENGTH = metadata["context_length"]
hidden_size = metadata["embedding_length"]

raw_q4k = {n for n, t in parser.tensors.items() if is_raw_q4k_weight(n, t)}
raw_q6k = {n for n, t in parser.tensors.items() if is_raw_q6k_weight(n, t)}
register_raw_q4k_weights(raw_q4k)
register_raw_q6k_weights(raw_q6k)

print("=== Construindo grafo de computação ===")
graph = QwenComputeGraphBuilder(metadata).build_compute_graph()


def read_hidden(tensor_mapping, name, n_rows):
    ptr = tensor_mapping[name]
    val = ptr.ptr if hasattr(ptr, 'ptr') else ptr
    buf = bytearray(n_rows * hidden_size * 2)
    hip.safe_memcpy_device_to_host(buf, ctypes.c_void_p(val), tag="output")
    return np.frombuffer(bytes(buf), dtype=np.float16).astype(np.float32).reshape(n_rows, hidden_size)


print("=== [EAGER, referência] BatchedFallbackExecutor(batch_size=4) ===")
mapper_eager = QwenTensorMapper(parser, metadata)
tm_eager = mapper_eager.map_and_allocate_tensors(
    allocator, hip, profiler=None, context_length=CONTEXT_LENGTH, batch_size=BATCH_SIZE
)
GGUFWeightLoader(model_path, parser, tm_eager).load_all(hip)
arena_eager = ActivationArena(allocator.allocate(4 * 1024 * 1024, "arena_eager", MemoryRegion.ACTIVATIONS))
executor_eager = BatchedFallbackExecutor(hip, allocator, arena_eager, graph, tm_eager, metadata, batch_size=BATCH_SIZE)

max_len = max(len(t) for t in PROMPTS_TOKENS)
for pos in range(max_len):
    tokens_at_pos = [PROMPTS_TOKENS[b][pos] for b in range(BATCH_SIZE)]
    executor_eager.decode_step_batch(tokens_at_pos, [pos] * BATCH_SIZE)

hs_eager = read_hidden(tm_eager, 'output_norm.output', BATCH_SIZE)
for b in range(BATCH_SIZE):
    print(f"  seq {b}: hidden[:4]={hs_eager[b][:4]}")

print(f"\n=== [HIP GRAPH] BatchedHIPGraphExecutor(batch_size={BATCH_SIZE}) ===")
mapper_graph = QwenTensorMapper(parser, metadata)
tm_graph = mapper_graph.map_and_allocate_tensors(
    allocator, hip, profiler=None, context_length=CONTEXT_LENGTH, batch_size=BATCH_SIZE
)
GGUFWeightLoader(model_path, parser, tm_graph).load_all(hip)
executor_graph = BatchedHIPGraphExecutor(hip, allocator, graph, tm_graph, metadata, batch_size=BATCH_SIZE)

for pos in range(max_len):
    tokens_at_pos = [PROMPTS_TOKENS[b][pos] for b in range(BATCH_SIZE)]
    executor_graph.execute_decode_batch(tokens_at_pos, [pos] * BATCH_SIZE)

hs_graph = read_hidden(tm_graph, 'output_norm.output', BATCH_SIZE)

print("\n=== Comparação ===")
worst_diff = 0.0
all_ok = True
for b in range(BATCH_SIZE):
    diff = np.max(np.abs(hs_graph[b] - hs_eager[b]))
    worst_diff = max(worst_diff, diff)
    ok = diff < 5e-3
    all_ok = all_ok and ok
    print(f"  seq {b}: diff={diff:.6f}  graph[:4]={hs_graph[b][:4]}  eager[:4]={hs_eager[b][:4]}  [{'OK' if ok else 'FALHOU'}]")

print(f"\nPior diff: {worst_diff:.6f}  ({'PASS' if all_ok else 'FALHOU'} @ tol 5e-3)")
print(">>> HIP GRAPH BATCHED CORRETO" if all_ok else ">>> DIVERGENCIA — investigar")

hip.cleanup()
