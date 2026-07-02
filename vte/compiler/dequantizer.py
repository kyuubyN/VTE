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


def to_fp16_bytes(raw: bytes, dtype: int, n_elements: int) -> bytes:
    """Converte os bytes crus de um tensor GGUF (qualquer dtype suportado) para FP16 contíguo."""
    if dtype == 0:  # F32
        arr = np.frombuffer(raw, dtype=np.float32, count=n_elements)
        return arr.astype(np.float16).tobytes()
    if dtype == 1:  # F16 (já no formato certo)
        return raw[:n_elements * 2]
    if dtype == 12:  # Q4_K
        arr = dequantize_q4_k(raw, n_elements)
        return arr.astype(np.float16).tobytes()
    if dtype == 14:  # Q6_K
        arr = dequantize_q6_k(raw, n_elements)
        return arr.astype(np.float16).tobytes()
    raise ValueError(f"Dtype GGUF não suportado para dequantização: {dtype}")
