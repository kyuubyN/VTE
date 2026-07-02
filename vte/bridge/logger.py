import logging
import sys
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    """
    Configura e retorna um logger seguro para o módulo VTE.
    
    Args:
        name: Nome do logger (geralmente __name__)
        
    Returns:
        Instância configurada do logger.
    """
    logger = logging.getLogger(name)
    
    if logger.hasHandlers():
        return logger
        
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    try:

        from vte.config import LOG_DIR
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / "hip_runtime.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:

        print(f"Aviso: Não foi possível configurar log para arquivo: {e}")
        
    return logger
