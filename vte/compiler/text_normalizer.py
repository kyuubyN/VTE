import unicodedata

def normalize(text: str) -> str:
    """
    Normaliza o texto para NFC (Normalization Form Canonical Composition) 
    conforme esperado pelo Qwen2.5 e outros modelos modernos.
    """
    return unicodedata.normalize("NFC", text.strip())
