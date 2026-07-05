"""
API principal do VTE - Estilo PyTorch/Ollama
"""

from pathlib import Path
from typing import Optional
from ..bridge.hip_runtime import HIPRuntime
from ..compiler.sanitizer import GGUFSanitizer
from .lifecycle import ModelLifecycleManager
from ..bridge.logger import get_logger
from ..compiler.tokenizer import QwenTokenizer
from ..core.sampler import Sampler
from ..core.hip_graph_executor import HIPGraphExecutor
from ..core.fallback_executor import FallbackExecutor
from ..compiler.qwen_mapper import QwenTensorMapper, ActivationArena
from ..bridge.memory import MemoryBlock, SlabAllocator, MemoryRegion
from vte.core.vram_profiler import VRAMProfiler
from ..compiler.gguf_parser import GGUFParser
from ..compiler.weight_loader import GGUFWeightLoader
from ..core.lm_head import LMHead
from ..core.incremental_decoder import IncrementalUTF8Decoder
import os
import time
import threading
import ctypes
import numpy as np

logger = get_logger("VTE.Model")

# vte/core/model.py -> raiz do repo é dois níveis acima. Resolvido a partir
# de __file__ (não do cwd) porque `vte-ui`/`vte` são instalados como pacote
# e podem ser invocados de qualquer diretório -- procurar "Model/" relativo
# ao cwd atual falhava silenciosamente sempre que o processo não era
# lançado a partir da raiz do projeto (ex.: atalho, outro terminal),
# mesmo com o arquivo do modelo presente e no lugar certo.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

class VTEModel:
    """
    Interface de alto nível para carregar e usar modelos de linguagem.
    
    Exemplo:
        >>> model = VTEModel.from_pretrained("qwen2.5:1.5b-q4_k_m")
        >>> response = model.generate("Olá", max_tokens=100)
    """
    
    MODEL_REGISTRY = {
        "qwen2.5:1.5b-q4_k_m": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "granite-4.1:3b-q8_0": "granite-4.1-3b-Q8_0.gguf",
        "qwen3.5:2b-q6_k": "Qwen3.5-2B-Q6_K.gguf",
    }
    
    DEFAULT_CONTEXT_LENGTH = 2048

    def __init__(self, gguf_path: str, use_hip_graph: bool = True, enable_fusion: bool = True, context_length: int = None, idle_timeout: int = 5, auto_unload: bool = True, max_batch_size: int = 1):
        self._path = gguf_path
        self._use_hip_graph = use_hip_graph
        self._enable_fusion = enable_fusion
        self._hip = None
        self._allocator = None
        self._graph = None
        self.compute_graph = None
        self._is_loaded = False
        self._context_length = context_length if context_length is not None else self.DEFAULT_CONTEXT_LENGTH
        # Fase II (prep): reserva headroom de VRAM no pool para permitir
        # generate_batch() alocar KV cache/arena de até `max_batch_size` sob
        # demanda, sem re-hipMalloc do pool inteiro. Pesos não escalam com
        # batch — só o KV cache/arena do maior batch pretendido precisa de
        # espaço reservado de antemão.
        self._max_batch_size = max(1, max_batch_size)
        
        self.idle_timeout = idle_timeout
        self.auto_unload = auto_unload
        self._last_access_time = 0
        self._watchdog_thread = None
        self._stop_watchdog = threading.Event()
        self.profiler = None
        
        self._lifecycle = ModelLifecycleManager(
            self,
            idle_timeout_seconds=idle_timeout,
            enable_auto_unload=auto_unload
        )
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        use_hip_graph: bool = True,
        enable_fusion: bool = True,
        idle_timeout_seconds: int = 300,
        enable_auto_unload: bool = True,
        max_batch_size: int = 1,
        context_length: int = None
    ) -> "VTEModel":
        if model_name not in cls.MODEL_REGISTRY:
            available = ", ".join(cls.MODEL_REGISTRY.keys())
            raise ValueError(
                f"Modelo '{model_name}' não encontrado.\n"
                f"Modelos disponíveis: {available}"
            )
        
        filename = cls.MODEL_REGISTRY[model_name]
        # Tenta primeiro relativo ao diretório de trabalho atual (permite
        # uma pasta Model/ própria fora do repo, se o usuário quiser rodar
        # assim), e só depois relativo à raiz do projeto -- que é o que
        # realmente resolve o caso comum de `vte-ui` ser instalado como
        # comando e invocado de qualquer lugar.
        candidates = [Path("Model") / filename, _REPO_ROOT / "Model" / filename]
        model_path = next((p for p in candidates if p.exists()), None)

        if model_path is None:
            checked = "\n".join(f"  - {p.resolve()}" for p in candidates)
            raise FileNotFoundError(
                f"Arquivo do modelo não encontrado. Caminhos checados:\n{checked}\n"
                f"Baixe o modelo e coloque-o na pasta 'Model/' na raiz do projeto "
                f"({_REPO_ROOT / 'Model'})."
            )
        
        instance = cls(
            str(model_path),
            use_hip_graph=use_hip_graph,
            enable_fusion=enable_fusion,
            context_length=context_length,
            idle_timeout=idle_timeout_seconds,
            auto_unload=enable_auto_unload,
            max_batch_size=max_batch_size
        )
        instance._load()
        instance._lifecycle.start_monitoring()
        return instance
    
    def _load_qwen2_metadata(self, sanitizer) -> dict:
        """Lê os hiperparâmetros REAIS do GGUF em vez de depender de defaults
        espalhados pelo código. Vários defaults estavam ERRADOS para o
        Qwen2.5-1.5B (head_count=12 não 16, rope.freq_base=1e6 não 1e4,
        eps=1e-6 não 1e-5), quebrando GQA, RoPE e lançando blocos de cabeça
        fora dos limites. As chaves aqui são as (sem prefixo 'qwen2.') que
        kernel_arg_builder/qwen_compute/qwen_mapper consultam. Comportamento
        INTOCADO em relação à versão anterior (pré-Granite) desta função."""
        from ..compiler.gguf_metadata import read_gguf_metadata
        raw = read_gguf_metadata(self._path, wanted_keys={
            "qwen2.embedding_length", "qwen2.block_count", "qwen2.context_length",
            "qwen2.attention.head_count", "qwen2.attention.head_count_kv",
            "qwen2.attention.key_length", "qwen2.feed_forward_length",
            "qwen2.rope.freq_base", "qwen2.attention.layer_norm_rms_epsilon",
        })

        embedding_length = raw.get("qwen2.embedding_length", sanitizer.header.embedding_length)
        head_count = raw.get("qwen2.attention.head_count", 12)
        # Qwen2.5 não armazena key_length separado no GGUF; head_dim = hidden / n_heads.
        head_dim = raw.get("qwen2.attention.key_length", embedding_length // head_count)

        metadata = {
            "embedding_length": embedding_length,
            "block_count": raw.get("qwen2.block_count", sanitizer.header.block_count),
            "context_length": raw.get("qwen2.context_length", sanitizer.header.context_length),
            "attention.head_count": head_count,
            "attention.head_count_kv": raw.get("qwen2.attention.head_count_kv", 2),
            "attention.key_length": head_dim,
            "feed_forward_length": raw.get("qwen2.feed_forward_length", 8960),
            "rope.freq_base": raw.get("qwen2.rope.freq_base", 1000000.0),
            "attention.layer_norm_rms_epsilon": raw.get("qwen2.attention.layer_norm_rms_epsilon", 1e-6),
        }
        logger.info(
            f"Hiperparâmetros GGUF (Qwen2.5): hidden={embedding_length}, heads={head_count}, "
            f"kv_heads={metadata['attention.head_count_kv']}, head_dim={head_dim}, "
            f"ffn={metadata['feed_forward_length']}, rope_theta={metadata['rope.freq_base']}, "
            f"eps={metadata['attention.layer_norm_rms_epsilon']:.2e}"
        )
        return metadata

    def _load_granite_metadata(self, sanitizer) -> dict:
        """Mesma ideia de `_load_qwen2_metadata`, mas para o Granite 4.1:
        lê hiperparâmetros reais do namespace 'granite.*' do GGUF, incluindo
        os 4 multiplicadores de escala (embedding_scale/attention_scale/
        residual_scale/logit_scale) que o Qwen2.5 não tem -- ausentes aqui
        eles ficam com o valor neutro (1.0, ou None para deixar o cálculo
        padrão de 1/sqrt(head_dim) assumir, no caso do attention_scale),
        que é como o resto do pipeline compartilhado (kernel_arg_builder,
        fallback_executor, etc.) já trata a ausência dessas chaves para o
        Qwen. Valores confirmados contra os bytes reais do GGUF, não
        adivinhados: attention.scale=0.015625, embedding_scale=12.0,
        residual_scale=0.22, logit_scale=10.0."""
        from ..compiler.gguf_metadata import read_gguf_metadata
        raw = read_gguf_metadata(self._path, wanted_keys={
            "granite.embedding_length", "granite.block_count", "granite.context_length",
            "granite.attention.head_count", "granite.attention.head_count_kv",
            "granite.rope.dimension_count", "granite.feed_forward_length",
            "granite.rope.freq_base", "granite.attention.layer_norm_rms_epsilon",
            "granite.attention.scale", "granite.embedding_scale",
            "granite.residual_scale", "granite.logit_scale",
        })

        embedding_length = raw.get("granite.embedding_length", sanitizer.header.embedding_length)
        head_count = raw.get("granite.attention.head_count", 40)
        head_dim = raw.get("granite.rope.dimension_count", embedding_length // head_count)

        metadata = {
            "embedding_length": embedding_length,
            "block_count": raw.get("granite.block_count", sanitizer.header.block_count),
            "context_length": raw.get("granite.context_length", sanitizer.header.context_length),
            "attention.head_count": head_count,
            "attention.head_count_kv": raw.get("granite.attention.head_count_kv", 8),
            "attention.key_length": head_dim,
            "feed_forward_length": raw.get("granite.feed_forward_length", 8192),
            "rope.freq_base": raw.get("granite.rope.freq_base", 1.0e7),
            "attention.layer_norm_rms_epsilon": raw.get("granite.attention.layer_norm_rms_epsilon", 1e-5),
            "attention_scale": raw.get("granite.attention.scale"),
            "embedding_scale": raw.get("granite.embedding_scale", 1.0),
            "residual_scale": raw.get("granite.residual_scale", 1.0),
            "logit_scale": raw.get("granite.logit_scale", 1.0),
            # 1 = NORM/intercalado (llama-model.cpp::llama_model_rope_type ->
            # LLAMA_ROPE_TYPE_NORM para LLM_ARCH_GRANITE) -- NÃO é o mesmo
            # NEOX/split-half do Qwen2.5 (que fica com o default 0). Não vem
            # do GGUF: é uma propriedade fixa da arquitetura, hardcoded pelo
            # próprio llama.cpp por arch, não por chave de metadado.
            "rope_type": 1,
        }
        logger.info(
            f"Hiperparâmetros GGUF (Granite): hidden={embedding_length}, heads={head_count}, "
            f"kv_heads={metadata['attention.head_count_kv']}, head_dim={head_dim}, "
            f"ffn={metadata['feed_forward_length']}, rope_theta={metadata['rope.freq_base']}, "
            f"eps={metadata['attention.layer_norm_rms_epsilon']:.2e}, "
            f"attention_scale={metadata['attention_scale']}, embedding_scale={metadata['embedding_scale']}, "
            f"residual_scale={metadata['residual_scale']}, logit_scale={metadata['logit_scale']}"
        )
        return metadata

    def _load_qwen3_5_metadata(self, sanitizer) -> dict:
        """Mesma ideia de `_load_qwen2_metadata`/`_load_granite_metadata`,
        pro Qwen 3.5 2B. Duas chaves NÃO vêm do GGUF (mesma situação do
        `rope_type=1` do Granite -- são propriedades fixas da arquitetura,
        confirmadas em config.json/modeling_qwen3_5.py reais, não em
        metadata do GGUF):
        - `rope_type=2` (NEOX parcial) e `rotary_dim=64`
          (partial_rotary_factor=0.25 * head_dim=256), só usados nas 6
          camadas `full_attention`.
        - As dimensões do Gated DeltaNet (`linear_attn.*`) também são fixas
          da arquitetura, não do GGUF -- vêm de `qwen3_5_mapper.py`."""
        from ..compiler.gguf_metadata import read_gguf_metadata
        from ..compiler.qwen3_5_mapper import (
            QWEN35_DEFAULT_HEAD_COUNT, QWEN35_DEFAULT_HEAD_COUNT_KV,
            QWEN35_DEFAULT_HEAD_DIM, QWEN35_DEFAULT_ROTARY_DIM, QWEN35_DEFAULT_FFN,
            QWEN35_DEFAULT_ROPE_THETA, QWEN35_LINEAR_NUM_HEADS,
            QWEN35_LINEAR_HEAD_K_DIM, QWEN35_LINEAR_HEAD_V_DIM,
            QWEN35_LINEAR_KEY_DIM, QWEN35_LINEAR_VALUE_DIM, QWEN35_CONV_DIM,
        )
        raw = read_gguf_metadata(self._path, wanted_keys={
            "qwen35.embedding_length", "qwen35.block_count", "qwen35.context_length",
            "qwen35.attention.head_count", "qwen35.attention.head_count_kv",
            "qwen35.attention.key_length", "qwen35.feed_forward_length",
            "qwen35.rope.freq_base", "qwen35.attention.layer_norm_rms_epsilon",
            "qwen35.rope.dimension_count",
        })

        embedding_length = raw.get("qwen35.embedding_length", sanitizer.header.embedding_length)
        head_count = raw.get("qwen35.attention.head_count", QWEN35_DEFAULT_HEAD_COUNT)
        head_dim = raw.get("qwen35.attention.key_length", QWEN35_DEFAULT_HEAD_DIM)
        rotary_dim = raw.get("qwen35.rope.dimension_count", QWEN35_DEFAULT_ROTARY_DIM)

        metadata = {
            "embedding_length": embedding_length,
            "block_count": raw.get("qwen35.block_count", sanitizer.header.block_count),
            "context_length": raw.get("qwen35.context_length", sanitizer.header.context_length),
            "attention.head_count": head_count,
            "attention.head_count_kv": raw.get("qwen35.attention.head_count_kv", QWEN35_DEFAULT_HEAD_COUNT_KV),
            "attention.key_length": head_dim,
            "feed_forward_length": raw.get("qwen35.feed_forward_length", QWEN35_DEFAULT_FFN),
            "rope.freq_base": raw.get("qwen35.rope.freq_base", QWEN35_DEFAULT_ROPE_THETA),
            "attention.layer_norm_rms_epsilon": raw.get("qwen35.attention.layer_norm_rms_epsilon", 1e-6),
            # RoPE parcial (Fase 1 do plano) -- fixo pela arquitetura, não
            # lido do GGUF (mesma natureza do rope_type=1 do Granite).
            "rope_type": 2,
            "rotary_dim": rotary_dim,
            # Gated DeltaNet (Fase 2/3) -- dimensões fixas da arquitetura.
            "linear_attn.num_heads": QWEN35_LINEAR_NUM_HEADS,
            "linear_attn.head_k_dim": QWEN35_LINEAR_HEAD_K_DIM,
            "linear_attn.head_v_dim": QWEN35_LINEAR_HEAD_V_DIM,
            "linear_attn.key_dim": QWEN35_LINEAR_KEY_DIM,
            "linear_attn.value_dim": QWEN35_LINEAR_VALUE_DIM,
            "linear_attn.conv_dim": QWEN35_CONV_DIM,
        }
        logger.info(
            f"Hiperparâmetros GGUF (Qwen3.5): hidden={embedding_length}, heads={head_count}, "
            f"kv_heads={metadata['attention.head_count_kv']}, head_dim={head_dim}, "
            f"rotary_dim={rotary_dim}, ffn={metadata['feed_forward_length']}, "
            f"rope_theta={metadata['rope.freq_base']}, "
            f"eps={metadata['attention.layer_norm_rms_epsilon']:.2e}"
        )
        return metadata

    def _load(self):
        sanitizer = GGUFSanitizer(Path(self._path))
        sanitizer.validate()

        # Arquitetura já foi validada pelo sanitizer (SUPPORTED_ARCHITECTURES)
        # -- aqui só escolhemos QUAL leitor de metadados/mapper/tokenizer usar,
        # nunca um terceiro caminho "desconhecido" (isso já teria sido
        # rejeitado por sanitizer.validate() antes de chegarmos aqui).
        architecture = sanitizer.header.architecture
        if architecture == "granite":
            metadata = self._load_granite_metadata(sanitizer)
        elif architecture == "qwen35":
            metadata = self._load_qwen3_5_metadata(sanitizer)
        else:
            metadata = self._load_qwen2_metadata(sanitizer)
        self.metadata = metadata
        self._architecture = architecture

        if self._hip is None:
            self._hip = HIPRuntime()
            self._hip.initialize()
        
        if self._allocator is None:
            vram_total = self._hip.get_device_properties()['total_global_mem']
            
            self.parser = GGUFParser(Path(self._path))
            self.parser.parse_tensors(sanitizer.header)

            # Etapa C: registra os pesos que ficam CRUS em Q4_K/Q6_K/Q8_0 a
            # partir da MESMA função usada pelo loader/mapper, para o executor
            # rotear esses nós ao gemv_* correto. Fonte única de verdade.
            from vte.compiler.qwen_mapper import is_raw_q4k_weight, is_raw_q6k_weight
            from vte.compiler.granite_mapper import is_raw_q8_0_weight
            from vte.core.fallback_executor import register_raw_q4k_weights, register_raw_q6k_weights, register_raw_q8_0_weights
            raw_q4k = {n for n, t in self.parser.tensors.items() if is_raw_q4k_weight(n, t)}
            raw_q6k = {n for n, t in self.parser.tensors.items() if is_raw_q6k_weight(n, t)}
            raw_q8_0 = {n for n, t in self.parser.tensors.items() if is_raw_q8_0_weight(n, t)}
            if architecture == "qwen35":
                # qwen_mapper.is_raw_q6k_weight (linha acima) é restrito por
                # NOME (ffn_down/token_embd) -- pensado pro GGUF misto
                # Q4_K/Q6_K do Qwen2.5. O GGUF do Qwen3.5 não mistura: TODA
                # matriz grande é Q6_K (attn_qkv, attn_gate, ffn_*,
                # token_embd), confirmado via gguf.GGUFReader real. Sem esta
                # união, attn_qkv/attn_gate/ffn_gate/ffn_up ficariam alocados
                # CRUS (qwen3_5_mapper.py) mas roteados como se fossem FP16
                # pelo executor -- corrupção silenciosa de dados. Aditivo:
                # não muda nada pro Qwen2.5/Granite (só roda quando
                # architecture=="qwen35").
                from vte.compiler.qwen3_5_mapper import is_raw_q6k_weight as qwen35_is_raw_q6k_weight
                raw_q6k |= {n for n, t in self.parser.tensors.items() if qwen35_is_raw_q6k_weight(n, t)}
            register_raw_q4k_weights(raw_q4k)
            register_raw_q6k_weights(raw_q6k)
            register_raw_q8_0_weights(raw_q8_0)
            logger.info(f"Pesos crus in-kernel: Q4_K={len(raw_q4k)} (gemv_q4k), Q6_K={len(raw_q6k)} (gemv_q6k), Q8_0={len(raw_q8_0)} (gemv_q8_0)")

            if architecture == "granite":
                from vte.compiler.granite_mapper import GraniteTensorMapper
                mapper = GraniteTensorMapper(self.parser, metadata)
            elif architecture == "qwen35":
                from vte.compiler.qwen3_5_mapper import Qwen3_5TensorMapper
                mapper = Qwen3_5TensorMapper(self.parser, metadata)
            else:
                mapper = QwenTensorMapper(self.parser, metadata)
            self._mapper = mapper  # Fase II (prep): reaproveitado por generate_batch()

            # Dimensiona o pool para o MAIOR batch_size pretendido (padrão 1 —
            # sem custo extra para quem só usa generate() de sequência única).
            # Pesos não escalam com batch; só o KV cache/arena do batch máximo
            # entram no cálculo de margem.
            reqs = mapper.calculate_memory_requirements(self._context_length, self._max_batch_size)
            requested_pool_size = reqs['with_margin']

            self._allocator = SlabAllocator(self._hip, vram_total, requested_pool_size=requested_pool_size)
            self._allocator.initialize()
            
            self.profiler = VRAMProfiler(self._allocator)
            
            self.tensor_mapping = mapper.map_and_allocate_tensors(self._allocator, self._hip, profiler=self.profiler, context_length=self._context_length)

            weight_loader = GGUFWeightLoader(
                self._path, self.parser, self.tensor_mapping,
                raw_q4k=raw_q4k, raw_q6k=raw_q6k, raw_q8_0=raw_q8_0,
            )
            loaded, total_bytes = weight_loader.load_all(self._hip)
            logger.info(f"Pesos injetados na VRAM: {loaded} tensores ({total_bytes / (1024*1024):.1f} MB)")

            logger.info("Construindo grafo de operações...")
            if architecture == "qwen35":
                # Grafo próprio (não QwenComputeGraphBuilder): as camadas do
                # Qwen3.5 NÃO têm todas a mesma sequência de nós (6
                # full_attention vs 18 linear_attention/Gated DeltaNet) --
                # diferente de Qwen2.5/Granite, que são estruturalmente
                # idênticos camada a camada (só os números mudam).
                from vte.compiler.qwen3_5_compute import Qwen3_5ComputeGraphBuilder
                compute_builder = Qwen3_5ComputeGraphBuilder(metadata)
            else:
                from vte.compiler.qwen_compute import QwenComputeGraphBuilder
                compute_builder = QwenComputeGraphBuilder(metadata)
            self.compute_graph = compute_builder.build_compute_graph()
            self._graph = self.compute_graph  # Alias para compatibilidade
            
            if self._enable_fusion:
                logger.info("Aplicando fusão de kernels...")
                from vte.compiler.fusion_analyzer import FusionAnalyzer
                from vte.compiler.fusion_applier import FusionApplier
                
                analyzer = FusionAnalyzer()
                candidates = analyzer.analyze(self.compute_graph)
                
                if candidates:
                    applier = FusionApplier()
                    self.compute_graph = applier.apply(self.compute_graph, candidates)
                    self._graph = self.compute_graph
                    logger.info(f"Fusão aplicada: {len(candidates)} mega-kernels criados")
                else:
                    logger.info("Nenhuma fusão possível encontrada")

            # Epilogue Fusion do residual: detecta GEMV -> Add(residual) e funde
            # o Add no epílogo do GEMV (mata ~56 launches/tok). Deve rodar após
            # o grafo final estar montado.
            from vte.core.fallback_executor import build_residual_fusion, RESIDUAL_FUSION
            build_residual_fusion(list(self._graph.nodes.values()))
            logger.info(f"Fusão de residual (epilogue): {len(RESIDUAL_FUSION)} GEMVs fundidos")

        if architecture == "granite":
            from vte.compiler.tokenizer import GraniteTokenizer
            self.tokenizer = GraniteTokenizer(gguf_path=self._path)
        elif architecture == "qwen35":
            from vte.compiler.tokenizer import Qwen3_5Tokenizer
            self.tokenizer = Qwen3_5Tokenizer(gguf_path=self._path)
        else:
            self.tokenizer = QwenTokenizer(gguf_path=self._path)
        self.sampler = Sampler()

        # Recupera o bloco de arena pré-alocado pelo mapper (Evita sobreposicao)
        arena_block = next((b for b in self._allocator.blocks if b.tag == "ACTIVATION_ARENA"), None)
        if not arena_block:
            raise RuntimeError("Bloco ACTIVATION_ARENA não encontrado no SlabAllocator após mapeamento.")
            
        # 2MB reservado para input_embeddings e input_ids no início da arena
        self.arena = ActivationArena(arena_block, start_offset=2097152)
        
        # Aloca buffers de ativação persistentes para o HIP Graph
        self._allocate_activation_buffers()

        lm_head_capture_info = None
        if self._use_hip_graph:
            # Resolve os dados do LM Head (peso/tied embeddings, buffer de
            # logits, kernel compilado) ANTES de construir o HIPGraphExecutor,
            # para poder gravá-lo DENTRO do mesmo grafo de decode (elimina o
            # lançamento eager separado a cada token, ~3.19ms/tok medidos).
            # Não dá para simplesmente criar o LMHead primeiro: LMHead.
            # compute_logits() reaproveita self.model.executor.codegen, que
            # ainda não existiria — usamos aqui uma instância própria de
            # CodegenEngine (o cache de kernel compilado é em disco, por hash
            # de conteúdo, não fica preso à instância).
            try:
                lm_head_capture_info = self._resolve_lm_head_capture_info()
            except Exception as e:
                logger.warning(f"Não foi possível pré-resolver o LM Head para captura no grafo: {e}. "
                                f"LM Head continuará rodando eager, fora do grafo.")
                lm_head_capture_info = None

        if self._use_hip_graph:
            try:
                self.executor = HIPGraphExecutor(self._hip, self._allocator, self._graph, self.tensor_mapping,
                                                  metadata=self.metadata, lm_head_info=lm_head_capture_info,
                                                  context_length=self._context_length)
                self.executor.build_decode_graph()
                logger.info(f"HIP Graph executor inicializado com sucesso (Nós: {len(self._graph.nodes)})")
            except Exception as e:
                logger.warning(f"HIP Graph falhou: {e}. Fazendo fallback para executor legado.")
                self.executor = FallbackExecutor(self._hip, self._allocator, self.arena, self._graph, self.tensor_mapping, metadata=self.metadata)
                self._use_hip_graph = False
                lm_head_capture_info = None
        else:
            logger.info("HIP Graphs desabilitado via flag. Usando FallbackExecutor.")
            self.executor = FallbackExecutor(self._hip, self._allocator, self.arena, self._graph, self.tensor_mapping, metadata=self.metadata)

        if lm_head_capture_info is not None:
            # HIP Graph capturou o LM Head: o LMHead reaproveita o MESMO
            # buffer/kernel já resolvidos, em vez de alocar/compilar de novo
            # (um segundo buffer não seria o que o grafo escreve).
            self.lm_head = LMHead(self, self._hip, self._allocator, tokenizer=self.tokenizer,
                                   logits_buffer=lm_head_capture_info['logits_buffer_ptr'],
                                   kernel_info={'kernel': lm_head_capture_info['kernel_fn'],
                                                'template': lm_head_capture_info['template']})
        else:
            self.lm_head = LMHead(self, self._hip, self._allocator, tokenizer=self.tokenizer)

        from vte.core.gpu_keepalive import GPUKeepAlive
        self._keepalive = GPUKeepAlive(self._hip, self._allocator)
        # Válvula de escape: o pulso fixo de 2ms foi medido como obsoleto (e
        # em 1.0ms, ativamente prejudicial -- picos de até 35ms, pior caso de
        # todos os testados) depois que o Split-KV encurtou o tick de decode
        # para ~9-10ms -- rápido o bastante para o DPM do WDDM nunca ter
        # tempo de derrubar o clock entre tokens. Ver README ("Bugs found
        # during development") para a medição completa (rajada contínua de
        # 200 tokens + gaps reais de 3s entre turnos, sem TDR em nenhum
        # caso). Mantido configurável via env var para reverter sem deploy
        # de código, caso um driver futuro ou hardware diferente reintroduza
        # o cenário que o pulso existia para prevenir.
        self._keepalive_pulse_s = float(os.environ.get("VTE_KEEPALIVE_PULSE_MS", "0.0")) / 1000.0

        self._is_loaded = True
    
    def _allocate_activation_buffers(self):
        """
        Aloca buffers persistentes na VRAM para as saídas intermediárias de cada nó.
        Necessário para o HIP Graph (grafo estático = endereços fixos).
        """
        from vte.compiler.ir import NodeType
        
        if not hasattr(self, 'compute_graph') or self.compute_graph is None:
            return
        
        allocated = 0
        for node in self.compute_graph.topological_sort():
            if node.op_type in [NodeType.INPUT, NodeType.OUTPUT]:
                continue
            
            out_name = node.output_tensor
            if not out_name or out_name in self.tensor_mapping:
                continue

            # Aloca PERSISTENTEMENTE a saída de todo nó de computação (não só as
            # saídas finais de camada). O HIP Graph exige endereços fixos para
            # TODOS os tensores intermediários referenciados durante a captura,
            # já que não há alocação dinâmica possível dentro de um grafo estático.
            #
            # Dimensionamos a dimensão dinâmica (-1, seq_len) como 1: esses buffers
            # persistentes são compartilhados por TODAS as capturas de grafo, e o
            # caso dominante (decode autoregressivo, um token por passo) sempre usa
            # seq_len=1. Dimensionar para context_length cheio custaria ~4.5GB só
            # nesses buffers (28 camadas x tensores intermediários), inviável na
            # VRAM disponível. Limitação conhecida: um prefill multi-token via
            # HIPGraphExecutor pode gravar além do buffer se seq_len > 1 nesses
            # tensores intermediários (não nas saídas finais persistentes, que já
            # existiam antes e continuam corretas) — acompanhar em trabalho futuro
            # de particionamento de buffers por "classe de shape" do grafo.
            size = 1
            for dim in node.shape:
                if dim > 0:
                    size *= dim
                elif dim == -1:
                    size *= 1
            size = size * 2  # fp16
            size = max(size, 512)
            
            block = self._allocator.allocate(size, f"act_{out_name}", MemoryRegion.ACTIVATIONS)
            self.tensor_mapping[out_name] = block.ptr
            allocated += 1

        logger.info(f"Buffers persistentes alocados: {allocated} tensores")

    def _resolve_lm_head_capture_info(self, batch_size: int = 1, tensor_mapping: dict = None) -> dict:
        """
        Resolve peso (com tied embeddings), aloca o buffer de logits e
        compila o kernel do LM Head — tudo ANTES do HIPGraphExecutor/
        BatchedHIPGraphExecutor existir, para que a captura do grafo de
        decode possa gravar o LM Head como mais um nó (mesmos requisitos de
        endereço fixo dos demais tensores). `batch_size` dimensiona o
        buffer de logits ([batch_size, vocab_size]) e o grid da captura
        (grid.y = batch_size); o peso e o kernel são os mesmos em qualquer
        batch_size, já que gemv_coalesced/gemv_q4k/gemv_q6k já suportam
        batch nativamente via blockIdx.y.

        Réplica deliberada da lógica de resolução em LMHead (nome do peso
        tied, critério gemv_q6k vs gemv_coalesced) — não dá para reaproveitar
        LMHead diretamente aqui porque LMHead.compute_logits() depende de
        self.model.executor.codegen, e o executor ainda não existe neste
        ponto. Usamos uma instância própria de CodegenEngine: o cache de
        kernel compilado é em disco, por hash do template renderizado, então
        não importa qual instância compila — o binário final é o mesmo.
        """
        from vte.compiler.codegen import CodegenEngine
        from vte.compiler.qwen_mapper import is_raw_q6k_weight
        from vte.compiler.granite_mapper import is_raw_q8_0_weight

        tensor_mapping = tensor_mapping if tensor_mapping is not None else self.tensor_mapping
        hidden_size = self.metadata.get('embedding_length', 1536)

        lm_head_name = 'output.weight'
        lm_head_ptr = tensor_mapping.get(lm_head_name)
        if lm_head_ptr is None:
            lm_head_name = 'token_embd.weight'
            lm_head_ptr = tensor_mapping.get(lm_head_name)
        if lm_head_ptr is None:
            raise ValueError("Peso do LM Head não encontrado em tensor_mapping (nem output.weight, nem token_embd.weight).")
        weight_ptr = lm_head_ptr.ptr if hasattr(lm_head_ptr, 'ptr') else lm_head_ptr

        tensor_info = self.parser.tensors.get(lm_head_name, {})
        weight_shape = tensor_info.get('shape')
        if not weight_shape:
            raise ValueError(f"Shape do peso do LM Head não encontrado ({lm_head_name}).")
        vocab_size = weight_shape[0]

        logits_block = self._allocator.allocate(
            size=batch_size * vocab_size * 2,  # FP16, [batch_size, vocab_size] -- tamanho fixo, nunca muda
            tag="logits_output" if batch_size == 1 else "logits_output_batch",
            region=MemoryRegion.SCRATCH
        )

        if is_raw_q6k_weight(lm_head_name, tensor_info):
            template = "gemv_q6k"
        elif is_raw_q8_0_weight(lm_head_name, tensor_info):
            template = "gemv_q8_0"
        else:
            template = "gemv_coalesced"
        codegen = CodegenEngine()
        hsaco_path = codegen.compile_kernel(
            template_name=template,
            arch=self._hip.get_gpu_architecture(),
            hidden_size=hidden_size,
            tile_size=256
        )
        _, kernel_fn = self._hip.load_kernel(hsaco_path, f"{template}_kernel")

        return {
            'weight_ptr': weight_ptr,
            'logits_buffer_ptr': logits_block.ptr,
            'kernel_fn': kernel_fn,
            'batch_size': batch_size,
            'template': template,
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
        }

    def _reset_gdn_state_if_needed(self):
        """Zera o estado persistente do Gated DeltaNet (Qwen3.5) + histórico
        do conv1d causal ANTES de cada nova chamada de generate().

        Bug real encontrado nesta sessão: `hipMemset` só era chamado UMA VEZ,
        em `map_and_allocate_tensors()` (carregamento do modelo) -- nunca de
        novo entre chamadas de generate(). Igual o KV cache (que reinicia
        implicitamente em kv_offset=0 a cada generate(), tratando cada
        chamada como uma sequência nova e independente), o estado do Gated
        DeltaNet também precisa reiniciar do zero a cada nova sequência --
        sem isto, uma segunda mensagem/geração na mesma instância do modelo
        carregado herdava o estado (contaminado) da geração ANTERIOR,
        explicando degradação que só aparece em conversas com mais de uma
        mensagem/chamada."""
        if getattr(self, '_architecture', None) != "qwen35":
            return
        from vte.compiler.qwen3_5_mapper import (
            QWEN35_FULL_ATTENTION_LAYERS, QWEN35_DEFAULT_LAYERS,
            QWEN35_LINEAR_NUM_HEADS, QWEN35_LINEAR_HEAD_K_DIM, QWEN35_LINEAR_HEAD_V_DIM,
            QWEN35_CONV_DIM, QWEN35_CONV_KERNEL_SIZE,
        )
        num_layers = self.metadata.get('block_count', QWEN35_DEFAULT_LAYERS)
        linear_attn_layers = sorted(set(range(num_layers)) - QWEN35_FULL_ATTENTION_LAYERS)
        state_size = QWEN35_LINEAR_NUM_HEADS * QWEN35_LINEAR_HEAD_K_DIM * QWEN35_LINEAR_HEAD_V_DIM * 4
        conv_hist_size = QWEN35_CONV_DIM * (QWEN35_CONV_KERNEL_SIZE - 1) * 4
        for l in linear_attn_layers:
            for key, size in ((f'blk.{l}.linear_attn_state', state_size),
                               (f'blk.{l}.conv1d_history', conv_hist_size)):
                ptr = self.tensor_mapping.get(key)
                if ptr is None:
                    continue
                pv = ptr.ptr if hasattr(ptr, 'ptr') else ptr
                self._hip.safe_memset(ctypes.c_void_p(pv), size, tag=key)

    def _default_repetition_penalty(self) -> float:
        """1.1 é o valor calibrado para Qwen2.5/Granite, mas é fraco demais
        para o Qwen3.5: em greedy decode (`temperature=0`) com textos mais
        longos, 1.1 deixa o modelo colapsar num loop (`**Cultura:** **Cultura:**...`
        até só `**` repetido); 1.3 evita o loop e termina naturalmente sem
        degenerar em ruído de pontuação (1.5 já super-corrige nessa direção).
        Medido comparando as três diretamente, mesmo prompt/seed."""
        return 1.3 if self._architecture == "qwen35" else 1.1

    def _default_temperature(self) -> float:
        """0.7 é o default global (sampling normal), mas o Qwen3.5 tem mais
        ruído numérico residual acumulado ao longo de 24 camadas do que
        Qwen2.5/Granite (ver docs/QWEN35.md) -- pequeno demais pra mudar QUAL
        token vence (por isso greedy/temperature=0 fica coerente), grande o
        bastante pra distorcer a cauda da distribuição de onde o sampling
        estocástico tira variedade. Testado 0.1/0.2/0.3/0.5/0.7: todo valor
        >0 degenerou (frases corridas sem pontuação, troca de idioma no meio
        da resposta); só 0.0 ficou consistentemente coerente. Trade-off
        aceito: respostas do Qwen3.5 ficam determinísticas (mesmo prompt =
        mesma resposta) até a causa do ruído em si ser reduzida."""
        return 0.0 if self._architecture == "qwen35" else 0.7

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = None,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = None,
    ):
        """Gera texto como um Generator, permitindo interrupção caller-side."""
        self._lifecycle.ensure_loaded()
        self._lifecycle.touch()

        if temperature is None:
            temperature = self._default_temperature()
        if repetition_penalty is None:
            repetition_penalty = self._default_repetition_penalty()

        self._reset_gdn_state_if_needed()

        input_tokens = self.tokenizer.encode(prompt)
        current_seq_len = len(input_tokens)

        # Prefill processa o prompt (posições 0..N-1). Ao final, output_norm.output
        # contém o hidden state da ÚLTIMA posição do prompt — é dele que sai a
        # predição do PRIMEIRO token gerado. Não reprocessamos o último token.
        #
        # No modo HIP Graph, processamos o prompt token a token reutilizando o
        # MESMO grafo de decode (seq_len=1, kv_offset variável) em vez do
        # antigo execute_prefill em lote (grafo seq_len=N): os buffers de
        # ativação persistentes são dimensionados para 1 posição, e um grafo
        # de N posições sofreria o mesmo overflow/corrupção que foi corrigido
        # no FallbackExecutor.prefill(). Como o grafo de decode já é
        # capturado uma única vez e reaproveitado (replay puro), processar o
        # prompt assim continua rápido — sem overhead de lançar 392 kernels
        # em Python por token do prompt.
        if self._use_hip_graph:
            for pos, tok in enumerate(input_tokens):
                self.executor.execute_decode(tok, kv_offset=pos)
        else:
            self.executor.prefill(input_tokens)

        # Fase 2 (LM Head no HIP Graph): quando o executor capturou o LM Head
        # dentro do próprio grafo de decode, o replay de execute_decode() já
        # escreveu os logits em lm_head.logits_buffer -- chamar
        # compute_logits() de novo aqui recalcularia a mesma GEMV eager,
        # duplicando o trabalho e anulando o ganho (~3.19ms/tok medidos).
        # Nesse caso, _read_logits só lê o que já está pronto na VRAM.
        lm_head_captured_in_graph = (
            self._use_hip_graph and getattr(self.executor, 'lm_head_info', None) is not None
        )

        def _read_logits():
            logits_buffer = bytearray(self.lm_head.vocab_size * 2)
            if lm_head_captured_in_graph:
                logits_ptr = self.lm_head.logits_buffer
            else:
                hidden_ptr = self.tensor_mapping.get('output_norm.output')
                hidden_val = hidden_ptr.ptr if hasattr(hidden_ptr, 'ptr') else hidden_ptr
                logits_ptr = self.lm_head.compute_logits(hidden_val, seq_len=1)
            # O LM head grava logits em FP16 (mesmo tipo do matmul_kernel).
            self._hip.safe_memcpy_device_to_host(
                logits_buffer, ctypes.c_void_p(logits_ptr), tag="logits_d2h"
            )
            return np.frombuffer(logits_buffer, dtype=np.float16).astype(np.float32)

        logits = _read_logits()

        stop_ids = self.tokenizer.stop_token_ids
        # Um token BPE byte-level pode carregar só METADE dos bytes de um
        # caractere multi-byte (emoji, acento) -- decodificar cada token
        # isoladamente cortava essas sequências no meio e produzia "�" (ver
        # vte/core/incremental_decoder.py). O decoder segura bytes
        # incompletos entre chamadas de feed() até a sequência completar.
        utf8_decoder = IncrementalUTF8Decoder()

        for i in range(max_tokens):
            next_token = self.sampler.sample(
                logits=logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                generated_tokens=input_tokens,
            )

            # Para no fim de turno (<|im_end|>) ou <|endoftext|>. Sem isto o
            # modelo continua gerando muito além da resposta, divergindo para
            # texto incoerente -- era o "texto quebrado no final". decode() já
            # retorna "" para esses tokens, mas o loop precisa PARAR, não só
            # emitir vazio.
            if next_token in stop_ids:
                break

            input_tokens.append(next_token)
            word = utf8_decoder.feed(self.tokenizer.decode_bytes([next_token]))
            if word:
                yield word

            current_seq_len += 1
            if current_seq_len >= self._context_length:
                break

            # Processa o token recém-gerado na sua posição para prever o próximo.
            if self._use_hip_graph:
                self.executor.execute_decode(next_token, kv_offset=current_seq_len - 1)
            else:
                self.executor.decode_step(next_token, current_seq_len - 1)

            # Histórico: 15ms fixos -> virou teto artificial de velocidade
            # (~66 tok/s) quando HIP Graph + fusão baixaram o tempo real de
            # GPU bem abaixo disso -> 2ms de pulso de keep-alive (proteção
            # contra o DPM do WDDM derrubar o clock entre tokens) -> 0.0ms
            # (padrão atual): com o Split-KV, o tick caiu para ~9-10ms,
            # rápido o bastante para o DPM nunca ter uma janela ociosa longa
            # o suficiente para agir — medido sem TDR em rajada contínua e em
            # gaps reais de 3s entre turnos (ver README). Configurável via
            # VTE_KEEPALIVE_PULSE_MS para reverter sem deploy de código.
            self._keepalive.pulse(self._keepalive_pulse_s)
            logits = _read_logits()

        tail = utf8_decoder.flush()
        if tail:
            yield tail

    def generate_batch(
        self,
        prompts: list,
        max_tokens: int = 100,
        temperature: float = None,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = None,
    ):
        """
        Fase II (preparação para scheduler): gera para `len(prompts)`
        sequências SIMULTANEAMENTE via BatchedHIPGraphExecutor — mesmos
        pesos do modelo já carregado (compartilhados, sem duplicar VRAM),
        KV cache/arena/RoPE cache alocados sob demanda para este batch_size.

        Escopo desta etapa (deliberado, ver Fase I/plano): todos os prompts
        precisam ter o MESMO número de tokens (lockstep puro). Suportar
        prompts de tamanhos diferentes exige mascaramento de atenção para
        ignorar posições de padding — isso é trabalho de scheduler real
        (Fase II completa: fila de admissão, paginação de KV cache), não
        uma mudança trivial de tokenização. Levantamos um erro claro em vez
        de gerar silenciosamente resultados incorretos.

        Yields:
            list[str]: uma palavra por sequência do batch, a cada "tick".
        """
        self._lifecycle.ensure_loaded()
        self._lifecycle.touch()

        if temperature is None:
            temperature = self._default_temperature()
        if repetition_penalty is None:
            repetition_penalty = self._default_repetition_penalty()

        batch_size = len(prompts)
        tokenized = [self.tokenizer.encode(p) for p in prompts]
        lengths = {len(t) for t in tokenized}
        if len(lengths) != 1:
            raise ValueError(
                f"generate_batch requer prompts do MESMO comprimento em tokens (lockstep) "
                f"nesta etapa — recebido comprimentos {sorted(lengths)}. Padding com máscara de "
                f"atenção fica para a Fase II completa (scheduler com admissão heterogênea)."
            )
        prompt_len = lengths.pop()

        from vte.core.batched_hip_graph_executor import BatchedHIPGraphExecutor
        batch_tensor_mapping = self._mapper.allocate_batch_runtime_state(
            self.tensor_mapping, self._allocator, self._hip,
            self._context_length, batch_size, profiler=self.profiler
        )

        # Mesmo padrão da Fase 2 (batch=1): resolve peso/buffer/kernel do LM
        # Head ANTES do BatchedHIPGraphExecutor existir, para gravá-lo DENTRO
        # do grafo batched (elimina o lançamento eager de compute_logits_batch,
        # ~7.5ms/tick medidos). Regra de ouro: o grafo só ESCREVE os logits na
        # VRAM -- o hipMemcpy D2H e o Sampler continuam rodando DEPOIS do
        # graph_launch retornar, nunca dentro da captura.
        lm_head_batch_info = None
        try:
            lm_head_batch_info = self._resolve_lm_head_capture_info(
                batch_size=batch_size, tensor_mapping=batch_tensor_mapping
            )
        except Exception as e:
            logger.warning(f"Não foi possível pré-resolver o LM Head batched para captura no grafo: {e}. "
                            f"LM Head batched continuará rodando eager, fora do grafo.")

        batch_executor = BatchedHIPGraphExecutor(
            self._hip, self._allocator, self._graph, batch_tensor_mapping, self.metadata,
            batch_size=batch_size, lm_head_info=lm_head_batch_info
        )

        for pos in range(prompt_len):
            tokens_at_pos = [tokenized[b][pos] for b in range(batch_size)]
            batch_executor.execute_decode_batch(tokens_at_pos, [pos] * batch_size)

        def _read_logits_batch():
            buf = bytearray(batch_size * self.lm_head.vocab_size * 2)
            if lm_head_batch_info is not None:
                logits_ptr = lm_head_batch_info['logits_buffer_ptr']
            else:
                hidden_ptr = batch_tensor_mapping['output_norm.output']
                hidden_val = hidden_ptr.ptr if hasattr(hidden_ptr, 'ptr') else hidden_ptr
                logits_ptr = self.lm_head.compute_logits_batch(hidden_val, batch_size)
            self._hip.safe_memcpy_device_to_host(buf, ctypes.c_void_p(logits_ptr), tag="logits_d2h")
            return np.frombuffer(bytes(buf), dtype=np.float16).astype(np.float32).reshape(batch_size, self.lm_head.vocab_size)

        logits_batch = _read_logits_batch()
        current_seq_len = prompt_len
        # Um decoder incremental POR sequência do batch -- cada uma tem seu
        # próprio buffer de bytes pendentes, independente das outras (ver
        # vte/core/incremental_decoder.py e o mesmo tratamento em generate()).
        utf8_decoders = [IncrementalUTF8Decoder() for _ in range(batch_size)]

        for i in range(max_tokens):
            next_tokens = []
            words = []
            for b in range(batch_size):
                tok = self.sampler.sample(
                    logits=logits_batch[b], temperature=temperature, top_p=top_p,
                    top_k=top_k, repetition_penalty=repetition_penalty,
                    generated_tokens=tokenized[b],
                )
                tokenized[b].append(tok)
                next_tokens.append(tok)
                words.append(utf8_decoders[b].feed(self.tokenizer.decode_bytes([tok])))
            yield words

            current_seq_len += 1
            if current_seq_len >= self._context_length:
                break

            batch_executor.execute_decode_batch(next_tokens, [current_seq_len - 1] * batch_size)
            self._keepalive.pulse(self._keepalive_pulse_s)
            logits_batch = _read_logits_batch()

        tails = [d.flush() for d in utf8_decoders]
        if any(tails):
            yield tails

    def unload(self):
        """Descarrega o modelo da VRAM (limpa slabs e libera HIP)."""
        logger.info("Iniciando unload seguro do modelo...")
        self._is_loaded = False
        self._lifecycle.unload()
        if self._allocator:
            self._allocator.cleanup()
            self._allocator = None
        if self._hip:
            self._hip.cleanup()
            self._hip = None
        logger.info("Modelo descarregado da VRAM com sucesso")

    def get_vram_usage(self) -> dict:
        """Retorna as estatísticas do VRAMProfiler"""
        if self.profiler:
            return self.profiler.get_summary_dict()
        return {'total_mb': 0, 'weights_mb': 0, 'kv_cache_mb': 0, 'arena_mb': 0, 'scratch_mb': 0}

    def get_model_status(self) -> dict:
        """Retorna status do modelo"""
        return self._lifecycle.get_status()
    
    def __del__(self):
        try:
            if hasattr(self, '_lifecycle') and self._lifecycle._is_loaded:
                logger.info("Cleanup automático: descarregando modelo...")
                self._lifecycle.unload()
        except Exception as e:
            logger.error(f"Erro no cleanup automático: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()
        return False
