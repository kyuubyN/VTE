from typing import Dict
from vte.core.model_config import ModelConfig
from vte.bridge.memory import SlabAllocator, MemoryBlock, MemoryRegion
from vte.bridge.errors import HIPSafetyError
from vte.compiler.ir import IRGraph, IRNode, FusedGateUpProjNode, QuantizationInfo
from vte.compiler.rope_computer import compute_rope_cache
from vte.bridge.logger import get_logger

logger = get_logger(__name__)

# IDs de tipo GGML relevantes.
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14


def is_raw_q4k_weight(name: str, tensor_info: dict) -> bool:
    """
    Fonte única de verdade: True se o tensor fica CRU em Q4_K na VRAM (roteado
    ao gemv_q4k). Precisa casar exatamente com o roteamento do executor.

    Cobertos: ffn_gate/ffn_up (uniformemente Q4_K) e os ffn_down que são Q4_K
    (o down_proj é misto Q4_K/Q6_K neste GGUF). attn_* seguem FP16 por ora.
    """
    if tensor_info.get('dtype') != GGML_TYPE_Q4_K:
        return False
    return (name.endswith("ffn_gate.weight")
            or name.endswith("ffn_up.weight")
            or name.endswith("ffn_down.weight"))


def is_raw_q6k_weight(name: str, tensor_info: dict) -> bool:
    """
    True se o tensor fica CRU em Q6_K na VRAM (roteado ao gemv_q6k / embedding
    lookup dequantizado).

    Cobertos: ffn_down (Q6_K, K=8960) e token_embd (Fase D.1 — tied embeddings:
    o mesmo tensor serve de peso do lm_head via gemv_q6k E do embedding lookup
    via embedding_lookup_q6k, dequantizado sob demanda em ambos os casos; sem
    isso o lm_head continuaria pagando ~277 MB de leitura extra por token).
    attn_v (também Q6_K) fica FP16 por ora.
    """
    if tensor_info.get('dtype') != GGML_TYPE_Q6_K:
        return False
    return name.endswith("ffn_down.weight") or name == "token_embd.weight"


class ActivationArena:
    """
    Arena Allocator para ativações efêmeras.
    
    O SlabAllocator aloca UM bloco gigante (ex: 50MB) na região ACTIVATIONS.
    Esta classe gerencia sub-alocações dentro desse bloco usando Bump Pointer.
    
    O reset (current_offset = 0) só deve ser chamado APÓS hipDeviceSynchronize(),
    garantindo que a GPU terminou de usar as ativações da camada anterior.
    """
    def __init__(self, slab_block: MemoryBlock, start_offset: int = 0):
        self.block = slab_block
        self.start_offset = start_offset
        self.current_offset = start_offset
        self._synchronized = True
    
    def allocate(self, size_bytes: int, alignment: int = 64) -> tuple[int, int]:
        """Retorna (absolute_ptr, offset_within_block)"""
        # A flag _synchronized é usada para avisar se a arena não foi resetada corretamente.
        # Nós podemos alocar múltiplos tensores no mesmo ciclo.
        
        aligned_offset = ((self.current_offset + alignment - 1) // alignment) * alignment
        
        if aligned_offset + size_bytes > self.block.size:
            raise HIPSafetyError(
                f"Arena esgotada: {aligned_offset + size_bytes} > {self.block.size}"
            )
        
        ptr = self.block.ptr + aligned_offset
        self.current_offset = aligned_offset + size_bytes
        self._synchronized = False
        
        return ptr, aligned_offset
    
    def reset_after_sync(self):
        """
        Reseta a arena APÓS confirmar que a GPU terminou (via hipDeviceSynchronize).
        Será chamado pelo dispatcher no final de cada camada.
        """
        self.current_offset = self.start_offset
        self._synchronized = True

class QwenTensorMapper:
    def __init__(self, parser, metadata: dict):
        self.parser = parser
        self.metadata = metadata
        
        # Cria um objeto dummy pra passar pro ModelConfig
        class DummyModel:
            pass
        dummy = DummyModel()
        dummy.metadata = metadata
        self.config = ModelConfig(dummy)

    def calculate_memory_requirements(self, context_length: int = 2048, batch_size: int = 1) -> dict:
        """Calcula a memória necessária para os pesos, cache e arena.

        Os pesos são dequantizados para FP16 no carregamento (weight_loader.py),
        então o espaço reservado usa o tamanho FP16 (elementos * 2 bytes), não o
        tamanho quantizado original do arquivo GGUF.

        Fase I (Batched Decode): batch_size>1 multiplica o KV cache e a
        Activation Arena — cada sequência do batch precisa de seu próprio
        espaço de K/V (elas divergem em conteúdo desde o 1o token gerado) e
        suas próprias ativações intermediárias ([batch, features] em vez de
        [1, features]). Os pesos NÃO multiplicam (são compartilhados/lidos
        1x por todas as sequências do batch — essa reutilização é exatamente
        a alavanca de throughput que a Etapa I.1 validou).
        """
        weights_total = sum(self._calculate_fp16_size(t, n) for n, t in self.parser.tensors.items())

        layers = self.metadata.get("block_count", 28)
        kv_heads = 2
        head_dim = 128
        kv_pool_size = layers * 2 * kv_heads * head_dim * 2 * context_length * batch_size

        rope_size = context_length * head_dim * 2

        ffn_intermediate_size = 8960
        arena_size = int((context_length * ffn_intermediate_size * 2) * 1.2) * batch_size

        buffers_size = 20 * 1024 * 1024

        total = weights_total + kv_pool_size + rope_size + arena_size + buffers_size
        logger.warning(f"DEBUG REQS: weights={weights_total / 1024**2} MB, kv={kv_pool_size / 1024**2} MB, rope={rope_size / 1024**2} MB, arena={arena_size / 1024**2} MB, buffers={buffers_size / 1024**2} MB, total={total / 1024**2} MB (batch_size={batch_size})")
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
        Tamanho em bytes a alocar na VRAM para o tensor.

        Etapa C: os pesos Q4_K roteados para o gemv_q4k ficam CRUS na VRAM
        (n_blocks * 144 bytes / 256 pesos) em vez de dequantizados para FP16.
        Todo o resto continua FP16 (2 bytes/elemento). O critério de "cru" tem
        que casar EXATAMENTE com o roteamento do executor (só gate/up hoje),
        senão um peso cru cairia num kernel FP16 e produziria lixo.
        """
        elements = 1
        for dim in tensor_info["shape"]:
            elements *= dim
        if is_raw_q4k_weight(name, tensor_info):
            return (elements // 256) * 144    # Q4_K cru
        if is_raw_q6k_weight(name, tensor_info):
            return (elements // 256) * 210    # Q6_K cru
        return elements * 2

    def map_and_allocate_tensors(self, allocator: SlabAllocator, hip_runtime, profiler=None, context_length=2048, batch_size=1) -> Dict[str, int]:
        """Mapeia os tensores para a VRAM através do SlabAllocator"""
        logger.info(f"Iniciando Mapeamento Fail-Fast e Alocação (batch_size={batch_size})")

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

            # Checagem de Tied Embeddings Output
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
            
        # Mapeia KV cache para cada camada
        layers = self.config.get('num_hidden_layers', 28)
        kv_heads = self.config.get('attention.head_count_kv', 2)
        head_dim = self.config.get('attention.key_length', 128)
        # IMPORTANTE: usa o parâmetro `context_length` recebido (o mesmo usado
        # para dimensionar kv_pool_size em calculate_memory_requirements),
        # NÃO o metadado "max_position_embeddings"/context_length do GGUF —
        # este último é o contexto NATIVO máximo do modelo (32768 no
        # Qwen2.5), muito maior que o context_length real de execução
        # (2048 por padrão). Usar o valor errado aqui faz o offset de cada
        # camada avançar 16x mais rápido que o pool realmente alocado,
        # escrevendo muito além do KV_CACHE_POOL e corrompendo a Activation
        # Arena e os buffers persistentes alocados logo em seguida.
        # layer_kv_size_per_batch é o tamanho (bytes) de UM dos dois (K ou V)
        # para UMA sequência do batch — fórmula inalterada da era batch=1.
        # Fase I: cada camada agora reserva batch_size vezes esse espaço para
        # K e para V (uma sequência não pode ler/escrever no cache da outra).
        layer_kv_size_per_batch = kv_heads * head_dim * 2 * context_length
        layer_kv_size = layer_kv_size_per_batch * batch_size  # bytes p/ K (ou V) de TODAS as sequências do batch

        # layer_kv_size é o tamanho de UM dos dois (K ou V) — cada camada usa
        # 2x esse tamanho (K + V). O bug original colocava V na METADE do
        # espaço de K (layer_kv_size // 2, sobrepondo com o próprio K) e
        # avançava o offset por camada usando só layer_kv_size (metade do
        # necessário), fazendo o cache K/V de camadas adjacentes se
        # sobreporem — corrompendo silenciosamente a atenção de toda camada
        # após a primeira.
        #
        # Layout de batch (Fase I): dentro do espaço de K (ou de V) de uma
        # camada, a sequência b do batch ocupa
        # [b*layer_kv_size_per_batch, (b+1)*layer_kv_size_per_batch). O
        # tensor_mapping guarda o ponteiro do INÍCIO do slot da sequência 0;
        # os kernels recebem `kv_batch_stride` (em elementos __half, não
        # bytes) e deslocam por `batch_idx * kv_batch_stride` — mesmo padrão
        # de "1 pool único, offset calculado" já usado para camadas/K-V.
        current_kv_offset = kv_ptr.ptr
        for l in range(layers):
            # K cache (início da sequência 0)
            tensor_mapping[f'blk.{l}.kv_cache.k'] = current_kv_offset
            # V cache (logo após todo o bloco de K de todas as sequências)
            tensor_mapping[f'blk.{l}.kv_cache.v'] = current_kv_offset + layer_kv_size
            current_kv_offset += layer_kv_size * 2

        # Stride de batch em ELEMENTOS __half (não bytes) — consistente com a
        # aritmética de ponteiro __half* usada dentro dos kernels de RoPE e
        # FlashAttention. batch_size=1 -> stride igual ao tamanho total do
        # cache por camada (irrelevante, nunca somado a um batch_idx>0).
        tensor_mapping['kv_batch_stride_elements'] = layer_kv_size_per_batch // 2
            
        # Activation Arena
        arena_size = reqs['arena']
        arena_ptr = allocator.allocate(arena_size, "ACTIVATION_ARENA", MemoryRegion.ACTIVATIONS)
        if profiler:
            profiler.track_allocation("Activation Arena", arena_size / (1024*1024), "ACTIVATIONS")
        
        # Fallback registers para testes e nós que esperam estas chaves
        tensor_mapping['input_embeddings'] = arena_ptr.ptr
        tensor_mapping['input_ids'] = arena_ptr.ptr + 1048576 # offset 1MB
            
        # Constrói RoPE Cache
        logger.info("Construindo RoPE Cache...")
        self._build_and_upload_rope_cache(tensor_mapping, allocator, hip_runtime, context_length)
        
        return tensor_mapping
        
    def allocate_batch_runtime_state(
        self, weight_tensor_mapping: Dict, allocator: SlabAllocator, hip_runtime,
        context_length: int, batch_size: int, profiler=None
    ) -> Dict[str, int]:
        """
        Fase II (prep): aloca APENAS o estado dependente de batch (KV cache,
        Activation Arena, RoPE cache) para `batch_size`, reaproveitando os
        PONTEIROS DE PESO já carregados em `weight_tensor_mapping` (pesos não
        mudam com o batch — são lidos 1x e compartilhados por todas as
        sequências, exatamente a alavanca de throughput da Fase I).

        Usado para servir geração batched a partir de um modelo já carregado
        (batch=1) sem duplicar pesos na VRAM — só o KV cache/arena/staging
        (pequenos: dezenas de MB mesmo em batch=4) são alocados de novo.

        CRÍTICO: copia só os ponteiros de PESO real do GGUF (chaves em
        `self.parser.tensors`) — NUNCA o dict inteiro. `weight_tensor_mapping`
        (o tensor_mapping de produção) também contém buffers de ativação
        intermediária (q_proj.output, attn_norm.output, input_ids, etc.) já
        dimensionados para 1 LINHA (batch=1) por
        VTEModel._allocate_activation_buffers(). Copiar essas chaves faria
        `allocate_batched_activation_buffers` pulá-las (por já existirem no
        dict), e sequências de batch_idx>0 escreveriam fora dos limites
        desses buffers de 1 linha — corrompendo tudo, exceto o batch_idx=0
        (cujo offset dentro do buffer errado ainda é 0, "por sorte" correto).
        """
        tensor_mapping = {
            name: ptr for name, ptr in weight_tensor_mapping.items()
            if name in self.parser.tensors
        }

        reqs = self.calculate_memory_requirements(context_length, batch_size)
        kv_pool_size = reqs['kv_cache']
        kv_ptr = allocator.allocate(kv_pool_size, f"KV_CACHE_POOL_batch{batch_size}", MemoryRegion.KV_CACHE)
        if profiler:
            profiler.track_allocation(f"KV Cache Pool (batch={batch_size})", kv_pool_size / (1024*1024), "KV_CACHE")

        layers = self.config.get('num_hidden_layers', 28)
        kv_heads = self.config.get('attention.head_count_kv', 2)
        head_dim = self.config.get('attention.key_length', 128)
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
        head_dim = 128
        for k, v in self.metadata.items():
            if k.endswith('attention.key_length'):
                head_dim = v
        rope_theta = self.metadata.get('rope_theta', 10000.0)
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
        logger.info(f"RoPE cache adicionado ao mapping: cos=0x{cos_ptr:016x}, sin=0x{sin_ptr:016x}")
