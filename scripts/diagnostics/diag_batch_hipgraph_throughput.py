"""
Etapa I.2 (extensão HIP Graph) — throughput agregado real via
BatchedHIPGraphExecutor. Um batch_size por processo (env VTE_BATCH_TP_SIZE),
para não acumular VRAM entre configs.
"""
import os
import glob
import time
import numpy as np
import ctypes
from pathlib import Path

from vte.compiler.sanitizer import GGUFSanitizer
from vte.compiler.gguf_parser import GGUFParser
from vte.compiler.qwen_mapper import QwenTensorMapper, is_raw_q4k_weight, is_raw_q6k_weight
from vte.compiler.qwen_compute import QwenComputeGraphBuilder
from vte.compiler.weight_loader import GGUFWeightLoader
from vte.bridge.hip_runtime import HIPRuntime
from vte.bridge.memory import SlabAllocator
from vte.core.fallback_executor import register_raw_q4k_weights, register_raw_q6k_weights
from vte.core.batched_hip_graph_executor import BatchedHIPGraphExecutor

BATCH_SIZE = int(os.environ.get("VTE_BATCH_TP_SIZE", "1"))
N_TOKENS = int(os.environ.get("VTE_BATCH_TP_TOKENS", "100"))
WARMUP_TOKENS = 5

model_path = Path(glob.glob("Model/*.gguf")[0])
hip = HIPRuntime(); hip.initialize()
vram_total = hip.get_device_properties()['total_global_mem']
allocator = SlabAllocator(hip, vram_total, requested_pool_size=int(vram_total * 0.6))
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

raw_q4k = {n for n, t in parser.tensors.items() if is_raw_q4k_weight(n, t)}
raw_q6k = {n for n, t in parser.tensors.items() if is_raw_q6k_weight(n, t)}
register_raw_q4k_weights(raw_q4k)
register_raw_q6k_weights(raw_q6k)

graph = QwenComputeGraphBuilder(metadata).build_compute_graph()

mapper = QwenTensorMapper(parser, metadata)
tensor_mapping = mapper.map_and_allocate_tensors(
    allocator, hip, profiler=None, context_length=CONTEXT_LENGTH, batch_size=BATCH_SIZE
)
GGUFWeightLoader(model_path, parser, tensor_mapping).load_all(hip)

executor = BatchedHIPGraphExecutor(hip, allocator, graph, tensor_mapping, metadata, batch_size=BATCH_SIZE)

rng = np.random.default_rng(0)
tokens_per_seq = [int(rng.integers(0, 100000)) for _ in range(BATCH_SIZE)]

start_cap = time.perf_counter()
executor.build_decode_graph()
capture_time = time.perf_counter() - start_cap

for pos in range(WARMUP_TOKENS):
    executor.execute_decode_batch(tokens_per_seq, [pos] * BATCH_SIZE)

hip.synchronize()
start = time.perf_counter()
for i in range(N_TOKENS):
    pos = WARMUP_TOKENS + i
    executor.execute_decode_batch(tokens_per_seq, [pos] * BATCH_SIZE)
hip.synchronize()
elapsed = time.perf_counter() - start

tokens_generated = N_TOKENS * BATCH_SIZE
tok_per_sec_aggregate = tokens_generated / elapsed
tok_per_sec_per_seq = N_TOKENS / elapsed

print(f"RESULT batch_size={BATCH_SIZE} captura={capture_time:.3f}s elapsed={elapsed:.3f}s "
      f"ticks={N_TOKENS} tok_agregado={tokens_generated} "
      f"tps_agregado={tok_per_sec_aggregate:.2f} tps_por_sequencia={tok_per_sec_per_seq:.2f}")

hip.cleanup()
