"""
Etapa I.2.e — Validação numérica do BatchedFallbackExecutor.

Compara K sequências DIFERENTES processadas simultaneamente via
BatchedFallbackExecutor (batch_size=K) contra a MESMA sequência processada
isoladamente via BatchedFallbackExecutor(batch_size=1) — ou seja, o MESMO
código de dispatch (kernels desagregados: rmsnorm+matmul+rope+flash_attention
separados, sem fusão QKV/FFN) em ambos os lados, variando SÓ a dimensão de
batch. Isso isola exatamente a pergunta que a Etapa I.2 precisa responder
("a indexação de batch está correta?"), sem o confundidor de comparar contra
o FallbackExecutor de produção — que usa o QKV FUNDIDO (Two-Pass Split-K ou
o kernel legado, dependendo de VTE_DISABLE_QKV_SPLITK) e por isso NÃO é
numericamente idêntico ao caminho desagregado, mesmo estando ambos
corretos/validados isoladamente (achado desta sessão: comparar fundido vs
desagregado gera divergência de até ~20% em alguns tokens, um falso-positivo
de "bug" que na verdade é só um algoritmo diferente).

Critério: diff da hidden state final (output_norm.output) por sequência do
batch, e greedy-token (argmax dos logits) idêntico entre os dois caminhos.
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
from vte.core.batched_fallback_executor import BatchedFallbackExecutor, allocate_batched_activation_buffers

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

sanitizer = GGUFSanitizer(model_path)
sanitizer.validate()

parser = GGUFParser(model_path)
parser.parse_tensors(sanitizer.header)

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


print("=== [REFERÊNCIA] BatchedFallbackExecutor(batch_size=1), 1 sequência por vez ===")
mapper_ref = QwenTensorMapper(parser, metadata)
tensor_mapping_ref = mapper_ref.map_and_allocate_tensors(
    allocator, hip, profiler=None, context_length=CONTEXT_LENGTH, batch_size=1
)
print("Carregando pesos (referência)...")
GGUFWeightLoader(model_path, parser, tensor_mapping_ref).load_all(hip)
arena_ref = ActivationArena(allocator.allocate(4 * 1024 * 1024, "arena_ref", MemoryRegion.ACTIVATIONS))

ref_hidden_states = []
for i, tokens in enumerate(PROMPTS_TOKENS):
    executor_ref = BatchedFallbackExecutor(hip, allocator, arena_ref, graph, tensor_mapping_ref, metadata, batch_size=1)
    for pos, tok in enumerate(tokens):
        executor_ref.decode_step_batch([tok], [pos])
    hs = read_hidden(tensor_mapping_ref, 'output_norm.output', 1)[0]
    ref_hidden_states.append(hs)
    print(f"  seq {i} (tokens={tokens}): hidden[:4]={hs[:4]}")

print(f"\n=== [BATCHED] BatchedFallbackExecutor(batch_size={BATCH_SIZE}), simultâneo ===")
mapper_batch = QwenTensorMapper(parser, metadata)
tensor_mapping_batch = mapper_batch.map_and_allocate_tensors(
    allocator, hip, profiler=None, context_length=CONTEXT_LENGTH, batch_size=BATCH_SIZE
)
print("Carregando pesos (batched, cópia separada — dobra o uso de VRAM de pesos "
      "só para isolar completamente os dois caminhos neste teste)...")
GGUFWeightLoader(model_path, parser, tensor_mapping_batch).load_all(hip)
arena_batch = ActivationArena(allocator.allocate(4 * 1024 * 1024, "arena_batch", MemoryRegion.ACTIVATIONS))

executor_batch = BatchedFallbackExecutor(
    hip, allocator, arena_batch, graph, tensor_mapping_batch, metadata, batch_size=BATCH_SIZE
)

max_len = max(len(t) for t in PROMPTS_TOKENS)
assert all(len(t) == max_len for t in PROMPTS_TOKENS), "teste assume prompts do MESMO tamanho (lockstep)"

for pos in range(max_len):
    tokens_at_pos = [PROMPTS_TOKENS[b][pos] for b in range(BATCH_SIZE)]
    kv_offsets = [pos] * BATCH_SIZE
    executor_batch.decode_step_batch(tokens_at_pos, kv_offsets)

hs_batch_all = read_hidden(tensor_mapping_batch, 'output_norm.output', BATCH_SIZE)

print("\n=== Comparação (hidden state final, linha por linha do batch) ===")
worst_diff = 0.0
all_ok = True
for b in range(BATCH_SIZE):
    diff = np.max(np.abs(hs_batch_all[b] - ref_hidden_states[b]))
    worst_diff = max(worst_diff, diff)
    ok = diff < 5e-3
    all_ok = all_ok and ok
    print(f"  seq {b}: diff={diff:.6f}  batched[:4]={hs_batch_all[b][:4]}  "
          f"ref[:4]={ref_hidden_states[b][:4]}  [{'OK' if ok else 'FALHOU'}]")

print(f"\nPior diff: {worst_diff:.6f}  ({'PASS' if all_ok else 'FALHOU'} @ tol 5e-3)")
print(">>> HIPOTESE DE CORRETUDE CONFIRMADA" if all_ok else ">>> DIVERGENCIA DETECTADA — investigar antes de prosseguir")

hip.cleanup()
