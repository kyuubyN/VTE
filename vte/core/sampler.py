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
    """Estrategias de sampling em CPU via Numpy"""
    
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
        
        # 1. Repetition penalty
        if generated_tokens and repetition_penalty != 1.0:
            for token in set(generated_tokens):
                if token < len(logits):
                    if logits[token] > 0:
                        logits[token] /= repetition_penalty
                    else:
                        logits[token] *= repetition_penalty
        
        # Se a temperature for 0, é um argmax (greedy decode) deterministico
        if temperature <= 0.0:
            return int(np.argmax(logits))
            
        # 2. Temperature scaling
        if temperature != 1.0:
            logits = logits / temperature
        
        # 3. Top-k filtering
        if top_k > 0:
            top_k = min(top_k, len(logits))
            # np.partition encontra o k-ésimo maior elemento
            kth_largest = np.partition(logits, -top_k)[-top_k]
            indices_to_remove = logits < kth_largest
            logits[indices_to_remove] = -np.inf
        
        # 4. Top-p (nucleus) filtering
        if 0.0 < top_p < 1.0:
            # Ordena decrescente
            sorted_indices = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_indices]
            
            # Calcula probabilidades do subset atual
            cumulative_probs = np.cumsum(stable_softmax(sorted_logits))
            
            # Remove tokens com probabilidade acumulada > top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # Mantém pelo menos o token mais provável
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
            sorted_indices_to_remove[0] = False
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -np.inf
        
        # 5. Softmax e sampling
        probs = stable_softmax(logits)
        
        # Ocasionalmente fp32 precision issues fazem a sum = 0.9999999
        probs = probs / np.sum(probs)
        
        next_token = np.random.choice(len(probs), p=probs)
        
        return int(next_token)
