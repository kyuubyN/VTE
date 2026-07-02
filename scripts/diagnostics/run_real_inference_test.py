import os
import sys
import time

# Adiciona o diretorio atual ao PYTHONPATH para importar vte
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from vte.core.model import VTEModel
from vte.bridge.logger import get_logger

logger = get_logger("RealInferenceTest")

def main():
    logger.info("=== Iniciando Teste de Inferência Real (Hardware-in-the-Loop) ===")
    
    # 1. Carregar o Modelo
    logger.info("Carregando Qwen2.5 1.5B GGUF na VRAM...")
    start_load = time.perf_counter()
    
    # timeout curto para testar auto-unload
    model = VTEModel.from_pretrained("qwen2.5:1.5b-q4_k_m", idle_timeout_seconds=5)
    
    load_time = time.perf_counter() - start_load
    logger.info(f"Modelo carregado e mapeado em {load_time:.2f}s")
    
    status = model.get_model_status()
    logger.info(f"Status do Modelo: {status}")
    
    # 2. Enviar instrução básica
    prompt = "Resuma em uma frase o que é Inteligência Artificial."
    logger.info(f"Prompt: '{prompt}'")
    
    logger.info("Iniciando geração...")
    
    start_gen = time.perf_counter()
    
    generator = model.generate(prompt, max_tokens=20, temperature=0.7)
    
    tokens = []
    first_token_time = None
    
    for word in generator:
        if first_token_time is None:
            first_token_time = time.perf_counter()
            ttft = (first_token_time - start_gen) * 1000
            logger.info(f"-> Time to First Token (TTFT): {ttft:.1f}ms")
            
        tokens.append(word)
        sys.stdout.write(word)
        sys.stdout.flush()
        
    print() # newline
    
    end_gen = time.perf_counter()
    total_time = end_gen - start_gen
    
    # 3. Coletar e exibir logs
    tokens_generated = len(tokens)
    tps = tokens_generated / total_time
    
    logger.info("=== Métricas de Desempenho ===")
    logger.info(f"Tokens Gerados : {tokens_generated}")
    logger.info(f"Tempo Total    : {total_time:.2f}s")
    logger.info(f"Throughput     : {tps:.1f} tokens/sec")
    
    # 4. Forçar ou aguardar descarregamento
    logger.info("=== Teste do Auto-Unload ===")
    logger.info("Aguardando 6 segundos sem uso para acionar o Watchdog de Descarregamento...")
    for i in range(1, 7):
        time.sleep(1)
        logger.info(f"[{i}s] ocioso...")
        
    status = model.get_model_status()
    logger.info(f"Status final pós Idle: {status}")
    
    logger.info("Finalizando script (o cleanup final rodará via __del__ caso precise).")

if __name__ == "__main__":
    main()
