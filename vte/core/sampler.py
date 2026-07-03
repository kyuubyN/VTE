# vte/core/sampler.py

import numpy as np
from typing import Optional, List

def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax numericamente estavel para evitar overflow com FP32 logits"""
    # Subtrai o maximo para evitar overflow no exp()
    logits_shifted = logits - np.max(logits)
    exp_logits = np.exp(logits_shifted)
    return exp_logits / np.sum(exp_logits)

class Sampler:
    """Estrategias de sampling em CPU via Numpy.

    Otimizacao (medida): a versao anterior rodava argsort/softmax/cumsum
    sobre o vocabulario inteiro (151936 elementos) mesmo depois do top_k
    ja ter marcado quase tudo como -inf -- 9.4ms/tok medidos, o maior
    componente isolado do tempo por token em producao (maior que as 28
    camadas do modelo na GPU). Esta versao filtra para o subconjunto do
    top_k (tipicamente ~50 elementos) ANTES de ordenar/softmax/amostrar,
    e vetoriza o repetition penalty (era um loop Python puro sobre os
    tokens gerados). Resultado bit-exato preservado no caminho greedy
    (temperature<=0); nos demais casos a distribuicao de probabilidade
    final e matematicamente identica -- so o RNG consome numeros aleatorios
    em ordem diferente por operar sobre um array menor.
    """

    @staticmethod
    def sample(
        logits: np.ndarray,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        generated_tokens: Optional[List[int]] = None
    ) -> int:
        """
        Amostra o proximo token da distribuicao de logits.
        """
        logits = logits.copy()

        # 1. Repetition penalty (vetorizado -- sem loop Python por token)
        if generated_tokens and repetition_penalty != 1.0:
            token_ids = np.fromiter(set(generated_tokens), dtype=np.int64)
            token_ids = token_ids[token_ids < len(logits)]
            if token_ids.size > 0:
                vals = logits[token_ids]
                logits[token_ids] = np.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)

        # Se a temperature for 0, é um argmax (greedy decode) deterministico
        if temperature <= 0.0:
            return int(np.argmax(logits))

        # 2. Temperature scaling
        if temperature != 1.0:
            logits = logits / temperature

        # 3. Top-k: reduz para os candidatos plausiveis ANTES de qualquer
        # ordenacao/softmax/amostragem -- np.argpartition e O(n) (particao
        # parcial), nao O(n log n) como uma ordenacao completa, e todo o
        # trabalho pesado a seguir passa a operar sobre ~top_k elementos
        # em vez do vocabulario inteiro.
        if top_k > 0:
            top_k = min(top_k, len(logits))
            top_indices = np.argpartition(logits, -top_k)[-top_k:]
        else:
            top_indices = np.arange(len(logits))

        candidate_logits = logits[top_indices]

        # 4. Top-p (nucleus) filtering -- agora sobre o subconjunto pequeno
        if 0.0 < top_p < 1.0:
            order = np.argsort(candidate_logits)[::-1]
            sorted_logits = candidate_logits[order]

            cumulative_probs = np.cumsum(stable_softmax(sorted_logits))

            remove_mask = cumulative_probs > top_p
            # Mantém pelo menos o token mais provável
            remove_mask[1:] = remove_mask[:-1]
            remove_mask[0] = False

            keep_order = order[~remove_mask]
            final_indices = top_indices[keep_order]
        else:
            final_indices = top_indices

        # 5. Softmax e sampling -- so sobre os candidatos finais
        final_logits = logits[final_indices]
        probs = stable_softmax(final_logits)

        # Ocasionalmente fp32 precision issues fazem a sum = 0.9999999
        probs = probs / np.sum(probs)

        chosen = np.random.choice(len(probs), p=probs)

        return int(final_indices[chosen])
