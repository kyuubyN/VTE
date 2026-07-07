import numpy as np

QK_K = 256


def _q4k_scale_min(scales: np.ndarray, j: int):
    """
    Réplica de get_scale_min_k4 do llama.cpp para Q4_K.

    `scales` tem shape (n_blocks, 12), int. Retorna (sc, m) como float32
    por bloco para o sub-bloco `j` (0..7).
    """
    if j < 4:
        sc = scales[:, j] & 63
        m = scales[:, j + 4] & 63
    else:
        # Atenção aos índices: os 2 bits altos de `sc` vêm de scales[j-4]>>6,
        # mas os de `m` vêm de scales[j]>>6 (q[j-0] no llama.cpp), NÃO de
        # scales[j-4]. Usar j-4 nos dois batia por coincidência quando os bits
        # eram iguais, mas corrompia parte dos pesos.
        sc = (scales[:, j + 4] & 0x0F) | ((scales[:, j - 4] >> 6) << 4)
        m = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    return sc.astype(np.float32), m.astype(np.float32)


def dequantize_q4_k(raw: bytes, n_elements: int) -> np.ndarray:
    """
    Dequantiza um tensor Q4_K (llama.cpp super-block format) para float32,
    seguindo EXATAMENTE o layout do llama.cpp:

    Layout do bloco (144 bytes / 256 elementos):
        d:      fp16 (2 bytes)  - escala super-block dos "scales"
        dmin:   fp16 (2 bytes)  - escala super-block dos "mins"
        scales: 12 bytes        - 8 pares (scale, min) de 6 bits empacotados
        qs:     128 bytes       - 256 pesos de 4 bits (2 por byte)

    Ordem de saída (a parte que estava errada antes): o super-bloco é
    percorrido em 4 grupos de 64 elementos. Cada grupo de 64 usa 32 bytes
    consecutivos de qs; os primeiros 32 elementos vêm dos nibbles BAIXOS
    (q & 0xF) desses 32 bytes com scale/min do sub-bloco is+0, e os 32
    seguintes vêm dos nibbles ALTOS (q >> 4) com scale/min do sub-bloco
    is+1. A versão anterior misturava os dois nibbles em metades de 16, o
    que produzia valores de magnitude plausível mas semanticamente errados
    (pesos embaralhados → saída incoerente).
    """
    n_blocks = n_elements // QK_K
    block_bytes = 2 + 2 + 12 + 128
    blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * block_bytes).reshape(n_blocks, block_bytes)

    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)[:, 0]
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)[:, 0]
    scales = blocks[:, 4:16].astype(np.int32)
    qs = blocks[:, 16:144]

    out = np.empty((n_blocks, QK_K), dtype=np.float32)

    for mblk in range(4):  # cada grupo de 64 elementos (j = 64*mblk)
        sc0, mn0 = _q4k_scale_min(scales, 2 * mblk + 0)
        sc1, mn1 = _q4k_scale_min(scales, 2 * mblk + 1)
        d1 = d * sc0; m1 = dmin * mn0
        d2 = d * sc1; m2 = dmin * mn1

        q = qs[:, 32 * mblk:32 * mblk + 32]
        lo = (q & 0x0F).astype(np.float32)
        hi = (q >> 4).astype(np.float32)

        out[:, 64 * mblk:64 * mblk + 32] = d1[:, None] * lo - m1[:, None]
        out[:, 64 * mblk + 32:64 * mblk + 64] = d2[:, None] * hi - m2[:, None]

    return out.reshape(-1)[:n_elements]


def dequantize_q5_k(raw: bytes, n_elements: int) -> np.ndarray:
    """
    Dequantiza um tensor Q5_K (llama.cpp super-block format) para float32.

    Achado ao testar um segundo GGUF do Qwen3.5 (mesmo modelo, conversor
    diferente -- Unsloth em vez de Bartowski): esse arquivo usa Q5_K pra
    attn_qkv/ssm_out das camadas linear_attention (36 tensores), tipo que
    nenhum dos GGUFs anteriores usava. Mesma mecânica de scale/min do Q4_K
    (reaproveita `_q4k_scale_min`), mais um 5º bit ("qh") por peso.

    Layout do bloco (176 bytes / 256 elementos):
        d:      fp16 (2 bytes)  - escala super-block dos "scales"
        dmin:   fp16 (2 bytes)  - escala super-block dos "mins"
        scales: 12 bytes        - 8 pares (scale, min) de 6 bits, IDÊNTICO ao Q4_K
        qh:     32 bytes        - 5º bit (mais significativo) de cada peso,
                 um bit por peso, REUTILIZADO pelos 4 grupos de 64 (o bit
                 testado desloca 2 posições a cada grupo -- u1/u2 abaixo)
        qs:     128 bytes       - nibbles baixos/altos, igual ao Q4_K

    w = d*sc*(nibble + (16 se o bit qh estiver setado, senão 0)) - dmin*mn,
    mesmo agrupamento de 64 elementos (4 grupos) e mesma extração de
    scale/min do Q4_K (`get_scale_min_k4` no llama.cpp).
    """
    n_blocks = n_elements // QK_K
    block_bytes = 2 + 2 + 12 + 32 + 128
    blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * block_bytes).reshape(n_blocks, block_bytes)

    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)[:, 0]
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)[:, 0]
    scales = blocks[:, 4:16].astype(np.int32)
    qh = blocks[:, 16:48]
    qs = blocks[:, 48:176]

    out = np.empty((n_blocks, QK_K), dtype=np.float32)

    for mblk in range(4):
        sc0, mn0 = _q4k_scale_min(scales, 2 * mblk + 0)
        sc1, mn1 = _q4k_scale_min(scales, 2 * mblk + 1)
        d1 = d * sc0; m1 = dmin * mn0
        d2 = d * sc1; m2 = dmin * mn1

        u1 = np.uint8(1 << (2 * mblk))
        u2 = np.uint8(2 << (2 * mblk))
        qh_bit1 = np.where((qh & u1) != 0, 16.0, 0.0).astype(np.float32)
        qh_bit2 = np.where((qh & u2) != 0, 16.0, 0.0).astype(np.float32)

        q = qs[:, 32 * mblk:32 * mblk + 32]
        lo = (q & 0x0F).astype(np.float32)
        hi = (q >> 4).astype(np.float32)

        out[:, 64 * mblk:64 * mblk + 32] = d1[:, None] * (lo + qh_bit1) - m1[:, None]
        out[:, 64 * mblk + 32:64 * mblk + 64] = d2[:, None] * (hi + qh_bit2) - m2[:, None]

    return out.reshape(-1)[:n_elements]


def dequantize_q6_k(raw: bytes, n_elements: int) -> np.ndarray:
    """
    Dequantiza um tensor Q6_K (llama.cpp super-block format) para float32.

    Layout do bloco (210 bytes / 256 elementos):
        ql:     128 bytes - 4 bits baixos de cada peso de 6 bits (2 por byte)
        qh:     64 bytes  - 2 bits altos de cada peso de 6 bits (4 por byte)
        scales: 16 bytes  - int8, uma por sub-bloco de 16 elementos
        d:      fp16 (2 bytes) - escala super-block
    """
    n_blocks = n_elements // QK_K
    block_bytes = 128 + 64 + 16 + 2
    blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * block_bytes).reshape(n_blocks, block_bytes)

    ql_all = blocks[:, 0:128]
    qh_all = blocks[:, 128:192]
    sc_all = blocks[:, 192:208].copy().view(np.int8)
    d = blocks[:, 208:210].copy().view(np.float16).astype(np.float32)[:, 0]

    out = np.empty((n_blocks, QK_K), dtype=np.float32)

    for half in range(2):
        ql = ql_all[:, half * 64:half * 64 + 64]
        qh = qh_all[:, half * 32:half * 32 + 32]
        sc = sc_all[:, half * 8:half * 8 + 8]

        for l in range(32):
            is_ = l // 16

            q1 = ((ql[:, l] & 0x0F) | (((qh[:, l] >> 0) & 3) << 4)).astype(np.int32) - 32
            q2 = ((ql[:, l + 32] & 0x0F) | (((qh[:, l] >> 2) & 3) << 4)).astype(np.int32) - 32
            q3 = ((ql[:, l] >> 4) | (((qh[:, l] >> 4) & 3) << 4)).astype(np.int32) - 32
            q4 = ((ql[:, l + 32] >> 4) | (((qh[:, l] >> 6) & 3) << 4)).astype(np.int32) - 32

            base = half * 128
            out[:, base + l] = d * sc[:, is_ + 0].astype(np.float32) * q1.astype(np.float32)
            out[:, base + l + 32] = d * sc[:, is_ + 2].astype(np.float32) * q2.astype(np.float32)
            out[:, base + l + 64] = d * sc[:, is_ + 4].astype(np.float32) * q3.astype(np.float32)
            out[:, base + l + 96] = d * sc[:, is_ + 6].astype(np.float32) * q4.astype(np.float32)

    return out.reshape(-1)[:n_elements]


def dequantize_q8_0(raw: bytes, n_elements: int) -> np.ndarray:
    """
    Dequantiza um tensor Q8_0 (llama.cpp block format) para float32.

    Layout do bloco (34 bytes / 32 elementos) -- confirmado empiricamente
    contra os bytes reais do GGUF do Granite (gguf.GGUFReader, comparando
    contra hipóteses concorrentes de tamanho de bloco), não a partir de
    documentação de terceiros:
        d:  fp16 (2 bytes) - escala do bloco
        qs: 32 bytes       - 32 pesos int8, SIMÉTRICOS (sem min/offset)

    Formato mais simples que Q4_K/Q6_K: um único fator de escala por bloco,
    sem hierarquia de scale-of-scale.
        w[i] = d * qs[i]
    """
    QK8_0 = 32
    n_blocks = n_elements // QK8_0
    block_bytes = 2 + 32
    blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * block_bytes).reshape(n_blocks, block_bytes)

    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)[:, 0]
    qs = blocks[:, 2:34].copy().view(np.int8).astype(np.float32)

    out = d[:, None] * qs
    return out.reshape(-1)[:n_elements]


def dequantize_q5_0(raw: bytes, n_elements: int) -> np.ndarray:
    """
    Dequantiza um tensor Q5_0 (llama.cpp block format) para float32.

    Achado ao adicionar o Qwen2.5 0.5B (draft model do speculative decoding,
    Fase 5): esse GGUF mistura Q5_0 (132 de 290 tensores) além de F32/Q8_0/
    Q6_K/Q4_K -- tipo nunca antes visto nos outros 3 modelos já suportados,
    então nem o cálculo de tamanho (gguf_parser.py) nem o dequant existiam
    pra ele. Sem isso, o parser calculava o tamanho do tensor como se fosse
    FP16 (fallback genérico) -- 2.9x maior que o real (22 bytes/32 elementos
    em vez de ~1.4 bytes/elemento), estourando o offset calculado dos
    tensores seguintes e disparando a barreira de segurança de "tensor além
    do arquivo" (`_validate_tensor_bounds`).

    Layout do bloco (22 bytes / 32 elementos), confirmado contra
    gguf.GGUFReader real:
        d:  fp16 (2 bytes)  - escala do bloco
        qh: uint32 (4 bytes) - 5º bit (mais significativo) de cada um dos
            32 pesos, um bit por peso
        qs: 16 bytes         - 32 nibbles de 4 bits (2 por byte)

    w[j]      = ((qs[j] & 0x0F) | (bit(qh, j)      << 4)) - 16,  d escalado
    w[j+16]   = ((qs[j] >> 4)   | (bit(qh, j + 16) << 4)) - 16,  d escalado
    para j em 0..15 -- mesma extração de bit alto (`qh`) e remontagem de
    nibble baixo/alto do llama.cpp (`dequantize_row_q5_0`).
    """
    QK5_0 = 32
    n_blocks = n_elements // QK5_0
    block_bytes = 2 + 4 + 16
    blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * block_bytes).reshape(n_blocks, block_bytes)

    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)[:, 0]
    qh = blocks[:, 2:6].copy().view(np.uint32)[:, 0]
    qs = blocks[:, 6:22]

    j = np.arange(16)
    xh_0 = ((qh[:, None] >> (j[None, :] + 0)) << 4).astype(np.uint32) & 0x10
    xh_1 = ((qh[:, None] >> (j[None, :] + 12))).astype(np.uint32) & 0x10

    x0 = ((qs & 0x0F).astype(np.int32) | xh_0.astype(np.int32)) - 16
    x1 = ((qs >> 4).astype(np.int32) | xh_1.astype(np.int32)) - 16

    out = np.empty((n_blocks, QK5_0), dtype=np.float32)
    out[:, 0:16] = x0 * d[:, None]
    out[:, 16:32] = x1 * d[:, None]
    return out.reshape(-1)[:n_elements]


def to_fp16_bytes(raw: bytes, dtype: int, n_elements: int) -> bytes:
    """Converte os bytes crus de um tensor GGUF (qualquer dtype suportado) para FP16 contíguo."""
    if dtype == 0:  # F32
        arr = np.frombuffer(raw, dtype=np.float32, count=n_elements)
        return arr.astype(np.float16).tobytes()
    if dtype == 1:  # F16 (já no formato certo)
        return raw[:n_elements * 2]
    if dtype == 8:  # Q8_0
        arr = dequantize_q8_0(raw, n_elements)
        return arr.astype(np.float16).tobytes()
    if dtype == 12:  # Q4_K
        arr = dequantize_q4_k(raw, n_elements)
        return arr.astype(np.float16).tobytes()
    if dtype == 14:  # Q6_K
        arr = dequantize_q6_k(raw, n_elements)
        return arr.astype(np.float16).tobytes()
    if dtype == 6:  # Q5_0
        arr = dequantize_q5_0(raw, n_elements)
        return arr.astype(np.float16).tobytes()
    if dtype == 13:  # Q5_K
        arr = dequantize_q5_k(raw, n_elements)
        return arr.astype(np.float16).tobytes()
    raise ValueError(f"Dtype GGUF não suportado para dequantização: {dtype}")
