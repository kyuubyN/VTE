from typing import Dict
from vte.core.model_config import ModelConfig
from vte.bridge.memory import SlabAllocator, MemoryRegion
from vte.bridge.errors import HIPSafetyError
from vte.compiler.qwen_mapper import ActivationArena  # genérico (bump-pointer arena), não específico do Qwen
from vte.bridge.logger import get_logger

logger = get_logger(__name__)

# Hiperparâmetros do Granite 4.1 3B (Q8_0), verificados contra o GGUF real
# (gguf.GGUFReader) e o código-fonte do llama.cpp (src/models/granite.cpp) —
# não copiados do Qwen. Ver plano em curious-roaming-quasar.md para a fonte
# de cada número.
GRANITE_DEFAULT_LAYERS = 40
GRANITE_DEFAULT_KV_HEADS = 8
GRANITE_DEFAULT_HEAD_DIM = 64
GRANITE_DEFAULT_FFN = 8192
GRANITE_DEFAULT_ROPE_THETA = 1.0e7

GGML_TYPE_Q8_0 = 8


def is_raw_q8_0_weight(name: str, tensor_info: dict) -> bool:
    """
    Fonte única de verdade: True se o tensor fica CRU em Q8_0 na VRAM
    (roteado ao gemv_q8_0 / embedding_lookup_q8_0). Precisa casar
    EXATAMENTE com o roteamento do executor -- mesmo contrato de
    is_raw_q4k_weight/is_raw_q6k_weight em qwen_mapper.py, mas vivendo aqui
    (arquivo próprio do Granite).

    NÃO cobre attn_q/attn_k/attn_v/attn_output: mesma restrição do Qwen
    (is_raw_q4k_weight/is_raw_q6k_weight também excluem attn_*).

    attn_q/k/v: os kernels FUNDIDOS de QKV (fused_norm_matmul_rope,
    split_k_qkv_pass1) leem `proj_weight` como `__half*` puro, sem nenhuma
    lógica de dequant Q8_0 -- roteá-los como crus produz NaN.

    attn_output: TESTADO e MEDIDO nesta sessão (não suposição) -- ao
    contrário de q/k/v, o_proj É seguro rotear pra gemv_q8_0 (nó MATMUL
    isolado, kernel já sabe desquantizar). Só que o gemv_q8_0 atual não é
    vetorizado (bloco de 34 bytes não alinha a 16B, sem uint4 como o
    gemv_coalesced), então o throughput MEDIU PIOR com attn_output cru
    (~32.4 tok/s) do que em FP16 (~36.3 tok/s) -- a redução de banda não
    compensou o overhead do loop escalar de desquant. Reverter para FP16
    até o gemv_q8_0 ganhar uma versão vetorizada.

    ffn_gate/ffn_up/ffn_down e token_embd usam kernels PRÓPRIOS
    (gemv_q8_0/embedding_lookup_q8_0) que sabem dequantizar, por isso
    ficam crus (ali o ganho de VRAM/banda é obrigatório, não opcional).
    """
    if tensor_info.get('dtype') != GGML_TYPE_Q8_0:
        return False
    return not (name.endswith("attn_q.weight") or name.endswith("attn_k.weight")
                or name.endswith("attn_v.weight") or name.endswith("attn_output.weight"))


class GraniteTensorMapper:
    """
    Espelha a interface pública de qwen_mapper.py::QwenTensorMapper, mas com
    os números do Granite 4.1 3B (40 camadas, 8 kv_heads, head_dim=64,
    FFN=8192, RoPE θ=1e7) em vez dos do Qwen2.5. `qwen_mapper.py` permanece
    sem nenhuma edição — arquivo próprio por arquitetura (ver "Decisão de
    design" no plano), zero risco de regressão no caminho já validado do
    Qwen.

    Fase 1: todos os pesos ficam CRUS em Q8_0 na VRAM (`is_raw_q8_0_weight`
    roteia 100% dos tensores, já que o arquivo inteiro é Q8_0) -- essencial
    para caber no orçamento de 95% de VRAM (ver Contexto do plano).
    """

    def __init__(self, parser, metadata: dict):
        self.parser = parser
        self.metadata = metadata

        class DummyModel:
            pass
        dummy = DummyModel()
        dummy.metadata = metadata
        self.config = ModelConfig(dummy)

    def calculate_memory_requirements(self, context_length: int = 2048, batch_size: int = 1) -> dict:
        """Calcula a memória necessária para os pesos, cache e arena.

        Mesma fórmula de QwenTensorMapper.calculate_memory_requirements —
        só os números de entrada mudam (40 camadas, 8 kv_heads, head_dim=64,
        FFN=8192). Ver docstring da versão do Qwen para o raciocínio de
        batch_size (pesos não escalam, KV cache/arena escalam).
        """
        weights_total = sum(self._calculate_fp16_size(t, n) for n, t in self.parser.tensors.items())

        layers = self.metadata.get("block_count", GRANITE_DEFAULT_LAYERS)
        kv_heads = self.metadata.get("attention.head_count_kv", GRANITE_DEFAULT_KV_HEADS)
        head_dim = self.metadata.get("attention.key_length", GRANITE_DEFAULT_HEAD_DIM)
        kv_pool_size = layers * 2 * kv_heads * head_dim * 2 * context_length * batch_size

        rope_size = context_length * head_dim * 2

        ffn_intermediate_size = self.metadata.get("feed_forward_length", GRANITE_DEFAULT_FFN)
        arena_size = int((context_length * ffn_intermediate_size * 2) * 1.2) * batch_size

        buffers_size = 20 * 1024 * 1024

        total = weights_total + kv_pool_size + rope_size + arena_size + buffers_size
        logger.warning(f"DEBUG REQS (Granite): weights={weights_total / 1024**2} MB, kv={kv_pool_size / 1024**2} MB, rope={rope_size / 1024**2} MB, arena={arena_size / 1024**2} MB, buffers={buffers_size / 1024**2} MB, total={total / 1024**2} MB (batch_size={batch_size})")
        return {
            'weights': weights_total,
            'kv_cache': kv_pool_size,
            'arena': arena_size,
            'rope': rope_size,
            'buffers': buffers_size,
            'total': total,
            'with_margin': int(total * 1.2)
        }

    def _calculate_tensor_size(self, tensor_info: dict) -> int:
        return tensor_info["size"]

    def _calculate_fp16_size(self, tensor_info: dict, name: str = "") -> int:
        """
        Tamanho em bytes a alocar na VRAM para o tensor. Pesos roteados como
        crus em Q8_0 (todos, no Granite) ocupam o tamanho cru do bloco
        (34 bytes / 32 elementos) em vez do FP16 dequantizado (2 bytes/
        elemento) -- é essa diferença que faz o modelo caber no orçamento
        de VRAM (ver Contexto do plano: ~3.4GB crus vs ~6.8GB em FP16).
        """
        elements = 1
        for dim in tensor_info["shape"]:
            elements *= dim
        if is_raw_q8_0_weight(name, tensor_info):
            return (elements // 32) * 34  # Q8_0 cru
        return elements * 2

    def map_and_allocate_tensors(self, allocator: SlabAllocator, hip_runtime, profiler=None, context_length=2048, batch_size=1) -> Dict[str, int]:
        """Mapeia os tensores para a VRAM através do SlabAllocator (mesmo fluxo de QwenTensorMapper)."""
        logger.info(f"[Granite] Iniciando Mapeamento Fail-Fast e Alocação (batch_size={batch_size})")

        reqs = self.calculate_memory_requirements(context_length, batch_size)
        total_required = reqs['total']
        free_vram = allocator.get_stats()['free_bytes']

        if total_required > free_vram:
            raise HIPSafetyError(
                f"OOM Preventivo: Modelo requer {total_required / (1024**3):.2f}GB, "
                f"mas o Slab tem apenas {free_vram / (1024**3):.2f}GB livres."
            )

        tensor_mapping = {}
        for name, t_info in self.parser.tensors.items():
            if t_info.get('is_tied', False):
                continue

            fp16_size = self._calculate_fp16_size(t_info, name)
            tensor_hash = f"{name}_{fp16_size}"

            # Granite usa embeddings amarrados (tied) — sem output.weight
            # separado no GGUF, igual ao caso já tratado no Qwen.
            if name == "output.weight":
                if t_info.get('is_tied', False) or t_info['size'] == 0:
                    tied_name = t_info.get('tied_to', "token_embd.weight")
                    if tied_name in tensor_mapping:
                        tensor_mapping[name] = tensor_mapping[tied_name]
                        logger.info(f"Tied embedding resolvido: {name} aponta para {tied_name}")
                        continue

            region = MemoryRegion.WEIGHTS
            if profiler:
                profiler.track_allocation(name, fp16_size / (1024 * 1024), "WEIGHTS")

            ptr = allocator.allocate(fp16_size, tag=tensor_hash, region=region)
            tensor_mapping[name] = ptr

        # KV Cache Pool
        kv_pool_size = reqs['kv_cache']
        kv_ptr = allocator.allocate(kv_pool_size, "KV_CACHE_POOL", MemoryRegion.KV_CACHE)
        if profiler:
            profiler.track_allocation("KV Cache Pool", kv_pool_size / (1024*1024), "KV_CACHE")

        layers = self.config.get('num_hidden_layers', GRANITE_DEFAULT_LAYERS)
        kv_heads = self.config.get('attention.head_count_kv', GRANITE_DEFAULT_KV_HEADS)
        head_dim = self.config.get('attention.key_length', GRANITE_DEFAULT_HEAD_DIM)
        layer_kv_size_per_batch = kv_heads * head_dim * 2 * context_length
        layer_kv_size = layer_kv_size_per_batch * batch_size

        current_kv_offset = kv_ptr.ptr
        for l in range(layers):
            tensor_mapping[f'blk.{l}.kv_cache.k'] = current_kv_offset
            tensor_mapping[f'blk.{l}.kv_cache.v'] = current_kv_offset + layer_kv_size
            current_kv_offset += layer_kv_size * 2

        tensor_mapping['kv_batch_stride_elements'] = layer_kv_size_per_batch // 2

        # Activation Arena
        arena_size = reqs['arena']
        arena_ptr = allocator.allocate(arena_size, "ACTIVATION_ARENA", MemoryRegion.ACTIVATIONS)
        if profiler:
            profiler.track_allocation("Activation Arena", arena_size / (1024*1024), "ACTIVATIONS")

        tensor_mapping['input_embeddings'] = arena_ptr.ptr
        tensor_mapping['input_ids'] = arena_ptr.ptr + 1048576  # offset 1MB

        logger.info("[Granite] Construindo RoPE Cache...")
        self._build_and_upload_rope_cache(tensor_mapping, allocator, hip_runtime, context_length)

        return tensor_mapping

    def allocate_batch_runtime_state(
        self, weight_tensor_mapping: Dict, allocator: SlabAllocator, hip_runtime,
        context_length: int, batch_size: int, profiler=None
    ) -> Dict[str, int]:
        """Mesma lógica de QwenTensorMapper.allocate_batch_runtime_state — reaproveita
        os ponteiros de peso já carregados, só realoca o estado dependente de batch."""
        tensor_mapping = {
            name: ptr for name, ptr in weight_tensor_mapping.items()
            if name in self.parser.tensors
        }

        reqs = self.calculate_memory_requirements(context_length, batch_size)
        kv_pool_size = reqs['kv_cache']
        kv_ptr = allocator.allocate(kv_pool_size, f"KV_CACHE_POOL_batch{batch_size}", MemoryRegion.KV_CACHE)
        if profiler:
            profiler.track_allocation(f"KV Cache Pool (batch={batch_size})", kv_pool_size / (1024*1024), "KV_CACHE")

        layers = self.config.get('num_hidden_layers', GRANITE_DEFAULT_LAYERS)
        kv_heads = self.config.get('attention.head_count_kv', GRANITE_DEFAULT_KV_HEADS)
        head_dim = self.config.get('attention.key_length', GRANITE_DEFAULT_HEAD_DIM)
        layer_kv_size_per_batch = kv_heads * head_dim * 2 * context_length
        layer_kv_size = layer_kv_size_per_batch * batch_size

        current_kv_offset = kv_ptr.ptr
        for l in range(layers):
            tensor_mapping[f'blk.{l}.kv_cache.k'] = current_kv_offset
            tensor_mapping[f'blk.{l}.kv_cache.v'] = current_kv_offset + layer_kv_size
            current_kv_offset += layer_kv_size * 2
        tensor_mapping['kv_batch_stride_elements'] = layer_kv_size_per_batch // 2

        arena_size = reqs['arena']
        arena_ptr = allocator.allocate(arena_size, f"ACTIVATION_ARENA_batch{batch_size}", MemoryRegion.ACTIVATIONS)
        if profiler:
            profiler.track_allocation(f"Activation Arena (batch={batch_size})", arena_size / (1024*1024), "ACTIVATIONS")
        tensor_mapping['input_embeddings'] = arena_ptr.ptr
        tensor_mapping['input_ids'] = arena_ptr.ptr + 1048576

        self._build_and_upload_rope_cache(tensor_mapping, allocator, hip_runtime, context_length)

        return tensor_mapping

    def _build_and_upload_rope_cache(self, tensor_mapping: dict, allocator, hip_runtime, context_length: int):
        from vte.compiler.rope_cache_builder import RoPECacheBuilder

        max_seq_len = context_length
        head_dim = GRANITE_DEFAULT_HEAD_DIM
        for k, v in self.metadata.items():
            if k.endswith('attention.key_length'):
                head_dim = v
        rope_theta = GRANITE_DEFAULT_ROPE_THETA
        for k, v in self.metadata.items():
            if k.endswith('rope.freq_base'):
                rope_theta = v

        builder = RoPECacheBuilder(
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            rope_theta=rope_theta
        )
        cos_cache, sin_cache = builder.build_cache()
        cos_ptr, sin_ptr = builder.upload_to_vram(cos_cache, sin_cache, hip_runtime, allocator)

        tensor_mapping['rope_cos'] = cos_ptr
        tensor_mapping['rope_sin'] = sin_ptr
        logger.info(f"[Granite] RoPE cache adicionado ao mapping: cos=0x{cos_ptr:016x}, sin=0x{sin_ptr:016x}")
