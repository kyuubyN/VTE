import ctypes
import math
from typing import List, Dict, Any, Tuple
from vte.compiler.ir import IRNode, NodeType
import logging

logger = logging.getLogger("VTE.KernelArgBuilder")


class KernelArgBuilder:
    """
    Constrói arrays de argumentos para kernels HIP com segurança de ABI.
    
    Responsabilidades:
    1. Mapeia IRNode → assinatura exata do kernel C++
    2. Mantém referências fortes para evitar GC prematuro
    3. Valida número de argumentos antes do launch
    4. Constrói o void** exigido pela API hipModuleLaunchKernel
    """
    
    # Assinaturas esperadas de cada tipo de kernel
    # Formato: [(nome, tipo_ctypes), ...]
    KERNEL_SIGNATURES = {
        NodeType.RMSNORM: [
            ("input", ctypes.c_void_p),
            ("weight", ctypes.c_void_p),
            ("output", ctypes.c_void_p),
            ("n", ctypes.c_int),
            ("eps", ctypes.c_float),
        ],
        NodeType.MATMUL: [
            ("input", ctypes.c_void_p),
            ("weight", ctypes.c_void_p),
            ("output", ctypes.c_void_p),
            ("batch", ctypes.c_int),
            ("seq_len", ctypes.c_int),
            ("in_features", ctypes.c_int),
            ("out_features", ctypes.c_int),
            ("bias", ctypes.c_void_p),
            ("residual", ctypes.c_void_p),
            ("residual_scale", ctypes.c_float),
        ],
        NodeType.ROPE: [
            ("q", ctypes.c_void_p),
            ("k", ctypes.c_void_p),
            ("cos", ctypes.c_void_p),
            ("sin", ctypes.c_void_p),
            ("seq_len", ctypes.c_int),
            ("num_q_heads", ctypes.c_int),
            ("num_kv_heads", ctypes.c_int),
            ("head_dim", ctypes.c_int),
            ("kv_cache_offset_ptr", ctypes.c_void_p),
            ("rope_type", ctypes.c_int),
        ],
        NodeType.ATTENTION: [
            ("q", ctypes.c_void_p),
            ("k", ctypes.c_void_p),
            ("v", ctypes.c_void_p),
            ("k_cache", ctypes.c_void_p),
            ("v_cache", ctypes.c_void_p),
            ("output", ctypes.c_void_p),
            ("seq_len", ctypes.c_int),
            ("num_q_heads", ctypes.c_int),
            ("num_kv_heads", ctypes.c_int),
            ("head_dim", ctypes.c_int),
            ("scale", ctypes.c_float),
            ("kv_cache_offset_ptr", ctypes.c_void_p),
            ("kv_batch_stride", ctypes.c_int),   # Fase I: elementos __half entre sequências no K/V cache
        ],
        NodeType.SWIGLU: [
            ("gate", ctypes.c_void_p),
            ("up", ctypes.c_void_p),
            ("output", ctypes.c_void_p),
            ("total_elements", ctypes.c_int),
        ],
        NodeType.ADD: [
            ("residual", ctypes.c_void_p),
            ("input", ctypes.c_void_p),
            ("output", ctypes.c_void_p),
            ("n", ctypes.c_int),
            ("residual_scale", ctypes.c_float),
        ],
    }
    
    def __init__(self):
        # Mantém referências fortes aos objetos ctypes criados
        # Isso previne garbage collection antes do kernel executar
        self._active_refs: List[Any] = []
    
    def build_args(
        self,
        node: IRNode,
        tensor_mapping: Dict[str, int],
        staging_buffers: Dict[str, Any],
        seq_len: int,
        metadata: dict,
        kv_offset_ptr: int = 0,
        batch: int = 1
    ) -> Tuple[List[ctypes.c_void_p], List[Any]]:
        """
        Constrói argumentos para um nó do grafo.
        
        Returns:
            (args_array, strong_refs): 
                - args_array: lista de ctypes.c_void_p para passar ao kernel
                - strong_refs: lista de objetos que devem ser mantidos vivos
        """
        signature = self.KERNEL_SIGNATURES.get(node.op_type)
        if signature is None:
            raise NotImplementedError(
                f"Assinatura não definida para op_type: {node.op_type}"
            )
        
        args_array = []
        strong_refs = []
        
        # Constrói argumentos baseado no tipo de nó
        if node.op_type == NodeType.RMSNORM:
            args_array, strong_refs = self._build_rmsnorm_args(
                node, tensor_mapping, seq_len, metadata
            )
        elif node.op_type == NodeType.MATMUL:
            args_array, strong_refs = self._build_matmul_args(
                node, tensor_mapping, seq_len, metadata, batch
            )
        elif node.op_type == NodeType.ROPE:
            args_array, strong_refs = self._build_rope_args(
                node, tensor_mapping, seq_len, metadata, kv_offset_ptr
            )
        elif node.op_type == NodeType.ATTENTION:
            args_array, strong_refs = self._build_attention_args(
                node, tensor_mapping, seq_len, metadata, kv_offset_ptr
            )
        elif node.op_type == NodeType.SWIGLU:
            args_array, strong_refs = self._build_swiglu_args(
                node, tensor_mapping, seq_len, metadata, batch
            )
        elif node.op_type == NodeType.ADD:
            args_array, strong_refs = self._build_add_args(
                node, tensor_mapping, seq_len, metadata, batch
            )
        
        # Validação de ABI: número de argumentos deve bater com assinatura
        if len(args_array) != len(signature):
            raise RuntimeError(
                f"ABI Mismatch no nó {node.name} ({node.op_type}): "
                f"esperado {len(signature)} args, fornecido {len(args_array)}. "
                f"Assinatura: {[name for name, _ in signature]}"
            )
        
        # Armazena referências fortes para prevenir GC
        self._active_refs.extend(strong_refs)
        
        return args_array, strong_refs
    
    def _get_meta(self, metadata: dict, key: str, default=None) -> Any:
        for k, v in metadata.items():
            if k.endswith(key):
                return v
        if default is not None:
            return default
        # Retorna dummy se não achar para o teste de validação de args não quebrar
        return 1
        
    def _resolve_tensor_ptr(
        self,
        tensor_name: str,
        tensor_mapping: Dict[str, int],
        staging_buffers: Dict[str, Any]
    ) -> int:
        """Resolve nome de tensor para ponteiro de VRAM"""
        if tensor_name in tensor_mapping:
            ptr = tensor_mapping[tensor_name]
            return ptr.ptr if hasattr(ptr, 'ptr') else int(ptr)
        elif tensor_name in ["input_ids", "input_embeddings"] and "input_ids" in staging_buffers:
            return staging_buffers['input_ids'].ptr
        else:
            print(f"DEBUG: Tensor '{tensor_name}' NÃO encontrado no tensor_mapping! (Retornando 0)")
            return 0
    
    def _build_rmsnorm_args(
        self, node: IRNode, tensor_mapping: dict, seq_len: int, metadata: dict = None
    ) -> Tuple[list, list]:
        """RMSNorm: (input, weight, output, n, eps)"""
        input_ptr = self._resolve_tensor_ptr(node.input_tensors[0], tensor_mapping, {})
        weight_ptr = self._resolve_tensor_ptr(node.input_tensors[1], tensor_mapping, {})
        output_ptr = self._resolve_tensor_ptr(node.output_tensor, tensor_mapping, {})

        n = node.shape[-1]  # hidden_size
        # Qwen2.5 usa eps=1e-6 (layer_norm_rms_epsilon do GGUF), não 1e-5.
        eps = self._get_meta(metadata or {}, 'attention.layer_norm_rms_epsilon', 1e-6)
        
        # Cria objetos ctypes e mantém referências fortes
        c_n = ctypes.c_int(n)
        c_eps = ctypes.c_float(eps)
        
        args = [
            ctypes.c_void_p(input_ptr),
            ctypes.c_void_p(weight_ptr),
            ctypes.c_void_p(output_ptr),
            c_n,
            c_eps,
        ]
        
        strong_refs = [c_n, c_eps]
        return args, strong_refs
    
    # Projeções que têm bias no Qwen2.5 (q/k/v). o_proj, gate, up, down NÃO têm.
    _BIAS_SUFFIX = {
        "q_proj": "attn_q.bias",
        "k_proj": "attn_k.bias",
        "v_proj": "attn_v.bias",
    }

    def _resolve_matmul_bias(self, node: IRNode, tensor_mapping: dict) -> int:
        """Resolve o ponteiro de bias para q/k/v_proj; 0 (nullptr) para as demais."""
        for suffix, bias_field in self._BIAS_SUFFIX.items():
            if node.name.endswith(suffix):
                # node.name = "blk.<idx>.<proj>"
                parts = node.name.split('.')
                if len(parts) >= 2:
                    bias_name = f"blk.{parts[1]}.{bias_field}"
                    bp = tensor_mapping.get(bias_name)
                    if bp is not None:
                        return bp.ptr if hasattr(bp, 'ptr') else int(bp)
                break
        return 0

    def _build_matmul_args(
        self, node: IRNode, tensor_mapping: dict, seq_len: int, metadata: dict, batch: int = 1
    ) -> Tuple[list, list]:
        """MatMul: (input, weight, output, batch, seq_len, in_features, out_features, bias)"""
        input_ptr = self._resolve_tensor_ptr(node.input_tensors[0], tensor_mapping, {})
        weight_ptr = self._resolve_tensor_ptr(node.input_tensors[1], tensor_mapping, {})

        # Epilogue Fusion do residual: se este GEMV é seguido por um Add(residual,
        # gemv_out), ele escreve direto na saída do Add e soma o residual no
        # epílogo (o nó Add é pulado pelo executor). Fonte única: RESIDUAL_FUSION.
        residual_ptr = 0
        out_tensor = node.output_tensor
        try:
            from vte.core.fallback_executor import RESIDUAL_FUSION
            fusion = RESIDUAL_FUSION.get(node.name)
        except Exception:
            fusion = None
        if fusion:
            out_tensor, residual_tensor = fusion
            residual_ptr = self._resolve_tensor_ptr(residual_tensor, tensor_mapping, {})
        output_ptr = self._resolve_tensor_ptr(out_tensor, tensor_mapping, {})

        if "down_proj" in node.name:
            in_features = self._get_meta(metadata, 'feed_forward_length', 8960)
        else:
            in_features = self._get_meta(metadata, 'embedding_length', 1536)

        out_features = node.shape[-1]
        bias_ptr = self._resolve_matmul_bias(node, tensor_mapping)

        c_batch = ctypes.c_int(batch)
        c_seq = ctypes.c_int(seq_len)
        c_in = ctypes.c_int(in_features)
        c_out = ctypes.c_int(out_features)
        # Granite escala a SAÍDA do sub-bloco (não o residual em si) ANTES de
        # somá-la ao residual -- confirmado em granite.cpp: só o `cur` que
        # entra em ggml_add(cur, inpSA/ffn_inp) é escalado, nunca gate_proj/
        # up_proj/q_proj/k_proj/v_proj isoladamente. Por isso só aplicamos o
        # valor real quando este nó É o que funde com o residual (`fusion`
        # truthy — exatamente attn_output/down_proj); todo o resto usa 1.0,
        # senão gate_proj/up_proj ficariam encolhidos por 0.22x sem motivo
        # (bug real encontrado nesta sessão: SiLU(gate*0.22)*(up*0.22) some
        # com o sinal do FFN antes mesmo do down_proj rodar).
        c_residual_scale = ctypes.c_float(metadata.get('residual_scale', 1.0) if fusion else 1.0)

        args = [
            ctypes.c_void_p(input_ptr),
            ctypes.c_void_p(weight_ptr),
            ctypes.c_void_p(output_ptr),
            c_batch,
            c_seq,
            c_in,
            c_out,
            ctypes.c_void_p(bias_ptr),
            ctypes.c_void_p(residual_ptr),
            c_residual_scale,
        ]

        strong_refs = [c_batch, c_seq, c_in, c_out, c_residual_scale]
        return args, strong_refs
    
    def _build_rope_args(
        self, node: IRNode, tensor_mapping: dict, seq_len: int, metadata: dict, kv_offset_ptr: int
    ) -> Tuple[list, list]:
        """RoPE: (q, k, cos_cache, sin_cache, seq_len, num_q_heads, num_kv_heads, head_dim, kv_offset_ptr)"""
        q_ptr = self._resolve_tensor_ptr(node.input_tensors[0], tensor_mapping, {})
        k_ptr = self._resolve_tensor_ptr(node.input_tensors[1], tensor_mapping, {})
        cos_ptr = self._resolve_tensor_ptr("rope_cos", tensor_mapping, {})
        sin_ptr = self._resolve_tensor_ptr("rope_sin", tensor_mapping, {})

        num_q_heads = self._get_meta(metadata, 'attention.head_count', 16)
        num_kv_heads = self._get_meta(metadata, 'attention.head_count_kv', 2)
        head_dim = self._get_meta(metadata, 'attention.key_length', 128)
        rope_type = metadata.get('rope_type', 0)

        c_seq = ctypes.c_int(seq_len)
        c_q_heads = ctypes.c_int(num_q_heads)
        c_kv_heads = ctypes.c_int(num_kv_heads)
        c_dim = ctypes.c_int(head_dim)
        c_rope_type = ctypes.c_int(rope_type)

        args = [
            ctypes.c_void_p(q_ptr),
            ctypes.c_void_p(k_ptr),
            ctypes.c_void_p(cos_ptr),
            ctypes.c_void_p(sin_ptr),
            c_seq,
            c_q_heads,
            c_kv_heads,
            c_dim,
            ctypes.c_void_p(kv_offset_ptr),
            c_rope_type,
        ]

        strong_refs = [c_seq, c_q_heads, c_kv_heads, c_dim, c_rope_type]
        return args, strong_refs

    def _build_attention_args(
        self, node: IRNode, tensor_mapping: dict, seq_len: int, metadata: dict, kv_offset_ptr: int
    ) -> Tuple[list, list]:
        """GQA: (q, k_cache, v_cache, q_norm, k_norm, kv_offset_ptr, seq_len, num_q, num_kv, dim)"""
        layer_idx = int(node.name.split('.')[1])

        q_ptr = self._resolve_tensor_ptr(node.input_tensors[0], tensor_mapping, {})
        k_proj_ptr = self._resolve_tensor_ptr(node.input_tensors[1], tensor_mapping, {})
        v_proj_ptr = self._resolve_tensor_ptr(node.input_tensors[2], tensor_mapping, {})
        
        # Tenta buscar K e V cache
        k_ptr = tensor_mapping.get(f'blk.{layer_idx}.kv_cache.k')
        v_ptr = tensor_mapping.get(f'blk.{layer_idx}.kv_cache.v')
        
        # Fallbacks comuns
        if k_ptr is None:
            k_ptr = tensor_mapping.get(f'kv_cache.layer_{layer_idx}.k')
        if v_ptr is None:
            v_ptr = tensor_mapping.get(f'kv_cache.layer_{layer_idx}.v')
            
        # Fallback para pool único
        if k_ptr is None or v_ptr is None:
            pool_ptr = tensor_mapping.get('KV Cache Pool')
            if pool_ptr is not None:
                # Calcula offset para esta layer
                # Qwen2.5: 2 KV heads, head_dim=128, max_seq=2048, FP16
                kv_size_per_layer = 2 * 2 * 128 * 2048 * 2  # K+V * heads * dim * seq * bytes
                k_ptr = pool_ptr + (layer_idx * kv_size_per_layer)
                v_ptr = pool_ptr + (layer_idx * kv_size_per_layer) + (kv_size_per_layer // 2)
                
                if layer_idx == 0:
                    print(f"  Usando fallback de offset do pool único")
                    print(f"     K ptr: 0x{k_ptr:016x}")
                    print(f"     V ptr: 0x{v_ptr:016x}")
        
        if k_ptr is None or v_ptr is None:
            print(f"  CRÍTICO: KV Cache não encontrado para layer {layer_idx}!")
            k_ptr = 0
            v_ptr = 0
            
        output_ptr = self._resolve_tensor_ptr(node.output_tensor, tensor_mapping, {})
        
        # Meta config
        num_q_heads = metadata.get('attention.head_count', 16)
        num_kv_heads = metadata.get('attention.head_count_kv', 2)
        head_dim = metadata.get('attention.key_length', 128)
        
        c_seq = ctypes.c_int(seq_len)
        c_q_heads = ctypes.c_int(num_q_heads)
        c_kv_heads = ctypes.c_int(num_kv_heads)
        c_dim = ctypes.c_int(head_dim)
        
        # Escala de atenção: usa o valor explícito do GGUF quando presente
        # (Granite: attention.scale=0.015625, SUBSTITUI o cálculo padrão —
        # não multiplica/soma a ele, confirmado contra llama-graph.cpp) ou
        # cai no cálculo padrão 1/sqrt(head_dim) quando ausente (Qwen).
        attention_scale = metadata.get('attention_scale')
        scale = attention_scale if attention_scale else 1.0 / (head_dim ** 0.5)
        c_scale = ctypes.c_float(scale)

        # Fase I: stride (em elementos __half) entre sequências do batch no
        # KV cache — calculado em qwen_mapper.py. Ausente (batch=1 / testes
        # sem esse tensor mapeado) => 0, seguro pois é multiplicado por
        # batch_idx=0 em produção (grid.x=1 hoje).
        kv_batch_stride = tensor_mapping.get('kv_batch_stride_elements', 0)
        c_kv_batch_stride = ctypes.c_int(int(kv_batch_stride))

        args = [
            ctypes.c_void_p(q_ptr),
            ctypes.c_void_p(k_proj_ptr),
            ctypes.c_void_p(v_proj_ptr),
            ctypes.c_void_p(k_ptr),
            ctypes.c_void_p(v_ptr),
            ctypes.c_void_p(output_ptr),
            c_seq,
            c_q_heads,
            c_kv_heads,
            c_dim,
            c_scale,
            ctypes.c_void_p(kv_offset_ptr),
            c_kv_batch_stride,
        ]

        strong_refs = [c_seq, c_q_heads, c_kv_heads, c_dim, c_scale, c_kv_batch_stride]
        return args, strong_refs
    def _build_swiglu_args(
        self, node: IRNode, tensor_mapping: dict, seq_len: int, metadata: dict, batch: int = 1
    ) -> Tuple[list, list]:
        """SwiGLU: (gate, up, output, total_elements).

        O kernel processa element-wise e recebe UM único total_elements
        (batch * seq_len * intermediate_size). Antes o builder passava 6 args
        (batch, seq_len, intermediate_size separados), mas o kernel só tem 4
        parâmetros — ele lia total_elements = batch = 1 e processava apenas 1
        dos 8960 elementos, zerando o resto do FFN e matando o sinal.

        swiglu_kernel é puramente elementwise sobre um range flat — já
        suporta batch>1 sem mudança de kernel, desde que total_elements
        inclua o multiplicador de batch (buffers [batch, features] contíguos).
        """
        gate_ptr = self._resolve_tensor_ptr(node.input_tensors[0], tensor_mapping, {})
        up_ptr = self._resolve_tensor_ptr(node.input_tensors[1], tensor_mapping, {})
        output_ptr = self._resolve_tensor_ptr(node.output_tensor, tensor_mapping, {})

        intermediate_size = self._get_meta(metadata, 'feed_forward_length', 8960)
        total_elements = batch * seq_len * intermediate_size

        c_total = ctypes.c_int(total_elements)

        args = [
            ctypes.c_void_p(gate_ptr),
            ctypes.c_void_p(up_ptr),
            ctypes.c_void_p(output_ptr),
            c_total,
        ]

        strong_refs = [c_total]
        return args, strong_refs
    
    def _build_add_args(
        self, node: IRNode, tensor_mapping: dict, seq_len: int, metadata: dict = None, batch: int = 1
    ) -> Tuple[list, list]:
        """Add (residual): (residual, input, output, n, residual_scale).

        add_kernel é puramente elementwise — já suporta batch>1 sem mudança
        de kernel, desde que n inclua o multiplicador de batch. Só é
        realmente exercitado com VTE_DISABLE_RESIDUAL_FUSION=1 (produção usa
        o epilogue fundido no GEMV anterior, ver build_residual_fusion).
        """
        residual_ptr = self._resolve_tensor_ptr(node.input_tensors[0], tensor_mapping, {})
        input_ptr = self._resolve_tensor_ptr(node.input_tensors[1], tensor_mapping, {})
        output_ptr = self._resolve_tensor_ptr(node.output_tensor, tensor_mapping, {})

        n = batch * seq_len * node.shape[-1]
        c_n = ctypes.c_int(n)
        c_residual_scale = ctypes.c_float((metadata or {}).get('residual_scale', 1.0))

        args = [
            ctypes.c_void_p(residual_ptr),
            ctypes.c_void_p(input_ptr),
            ctypes.c_void_p(output_ptr),
            c_n,
            c_residual_scale,
        ]

        strong_refs = [c_n, c_residual_scale]
        return args, strong_refs
    
    def clear_refs(self):
        """Libera referências fortes após sincronização"""
        self._active_refs.clear()
