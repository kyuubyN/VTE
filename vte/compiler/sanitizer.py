import struct
import hashlib
from pathlib import Path
from dataclasses import dataclass
from vte.bridge.errors import HIPSafetyError
from vte.bridge.logger import get_logger

logger = get_logger(__name__)

@dataclass
class GGUFHeader:
    magic: bytes
    version: int
    tensor_count: int
    kv_count: int
    architecture: str = ""
    context_length: int = 0
    embedding_length: int = 0
    block_count: int = 0

class GGUFSanitizer:
    def __init__(self, model_path: str | Path):
        self.path = Path(model_path)
        self.header: GGUFHeader | None = None
        
    def _validate_or_generate_hash(self) -> bool:
        """
        Se QWEN2_5_EXPECTED_HASH estiver vazio em config.py, 
        calcula o hash e exibe para o usuário configurar.
        """
        from vte.config import QWEN2_5_EXPECTED_HASH
        
        calculated_hash = self._calculate_sha256()
        
        if not QWEN2_5_EXPECTED_HASH or QWEN2_5_EXPECTED_HASH == "sha256:PLACEHOLDER":
            logger.warning(f"PRIMEIRA EXECUÇÃO - Hash não configurado. Adicione no config.py: QWEN2_5_EXPECTED_HASH = '{calculated_hash}'")
            return True
            
        if calculated_hash != QWEN2_5_EXPECTED_HASH:
            raise HIPSafetyError(
                f"Hash não corresponde!\n"
                f"Calculado: {calculated_hash}\n"
                f"Esperado:  {QWEN2_5_EXPECTED_HASH}"
            )
            
        return True

    def _calculate_sha256(self) -> str:
        """Calcula o SHA-256 do arquivo em chunks de 8MB para não estourar a memória."""
        sha256_hash = hashlib.sha256()
        try:
            with open(self.path, "rb") as f:
                for byte_block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except IOError as e:
            raise HIPSafetyError(f"Falha ao ler arquivo para hash: {e}")

    def _read_string(self, f) -> str:
        length_bytes = f.read(8)
        if len(length_bytes) < 8:
            raise HIPSafetyError("EOF inesperado lendo string length")
        length = struct.unpack("<Q", length_bytes)[0]
        if length > 65536:
             raise HIPSafetyError(f"String excessivamente longa nos metadados: {length} bytes")
        str_bytes = f.read(length)
        if len(str_bytes) < length:
             raise HIPSafetyError("EOF inesperado lendo string data")
        return str_bytes.decode('utf-8', errors='replace')

    def _parse_kv_pairs(self, f, kv_count: int):
        for i in range(kv_count):
            key = self._read_string(f)

            vtype_bytes = f.read(4)
            if len(vtype_bytes) < 4:
                raise HIPSafetyError("EOF lendo value type")
            vtype = struct.unpack("<I", vtype_bytes)[0]
            
            if key == "general.architecture":
                if vtype == 8:
                    self.header.architecture = self._read_string(f)
                else:
                    raise HIPSafetyError("general.architecture não é string")
            elif key == "qwen2.context_length":
                if vtype == 4:
                    self.header.context_length = struct.unpack("<I", f.read(4))[0]
                else:
                    self._skip_value(f, vtype)
            elif key == "qwen2.embedding_length":
                if vtype == 4:
                     self.header.embedding_length = struct.unpack("<I", f.read(4))[0]
                else:
                     self._skip_value(f, vtype)
            elif key == "qwen2.block_count":
                if vtype == 4:
                     self.header.block_count = struct.unpack("<I", f.read(4))[0]
                else:
                     self._skip_value(f, vtype)
            else:
                self._skip_value(f, vtype)

    def _skip_value(self, f, vtype: int):

        sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}
        if vtype in sizes:
            f.read(sizes[vtype])
        elif vtype == 8:
            self._read_string(f)
        elif vtype == 9:
             atype_bytes = f.read(4)
             alen_bytes = f.read(8)
             if len(atype_bytes) < 4 or len(alen_bytes) < 8:
                 raise HIPSafetyError("EOF lendo array metadata")
             atype = struct.unpack("<I", atype_bytes)[0]
             alen = struct.unpack("<Q", alen_bytes)[0]
             if alen > 250000:
                 raise HIPSafetyError(f"Array muito longo nos metadados: {alen} elementos")
             for _ in range(alen):
                 self._skip_value(f, atype)
        else:
             raise HIPSafetyError(f"Tipo de valor GGUF não suportado/conhecido: {vtype}")

    def _validate_metadata_consistency(self):
        """Valida que metadados são consistentes entre si"""
        if self.header.context_length > 100000:
            raise HIPSafetyError(f"context_length suspeito: {self.header.context_length}")
            
        if self.header.architecture != "qwen2":
            raise HIPSafetyError(f"Arquitetura não suportada. Esperado: qwen2, Recebido: {self.header.architecture}")
            
        if self.header.block_count != 28:
            raise HIPSafetyError(f"Block count inválido para Qwen2.5-1.5B. Esperado: 28, Recebido: {self.header.block_count}")

    def validate(self) -> bool:
        logger.info(f"Validando modelo: {self.path}")
        
        if not self.path.exists() or not self.path.is_file():
            raise HIPSafetyError("Modelo não encontrado")
            
        if self.path.name != "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf":
            raise HIPSafetyError("Nome do modelo incorreto")
            
        from vte.config import ALLOWED_MODEL_SIZE_MIN, ALLOWED_MODEL_SIZE_MAX
        size = self.path.stat().st_size
        if size < ALLOWED_MODEL_SIZE_MIN or size > ALLOWED_MODEL_SIZE_MAX:
             raise HIPSafetyError(f"Tamanho fora do esperado: {size} bytes")
             
        self._validate_or_generate_hash()
        
        try:
            with open(self.path, "rb") as f:
                magic = f.read(4)
                if magic != b"GGUF":
                    raise HIPSafetyError("Magic number inválido (não é arquivo GGUF)")
                    
                version = struct.unpack("<I", f.read(4))[0]
                if version != 3:
                    raise HIPSafetyError(f"Versão GGUF não suportada: {version}. Somente v3 permitida.")
                    
                tensor_count = struct.unpack("<Q", f.read(8))[0]
                if tensor_count > 1000:
                    raise HIPSafetyError(f"Tensor count excessivo: {tensor_count}")
                    
                kv_count = struct.unpack("<Q", f.read(8))[0]
                if kv_count > 100:
                     raise HIPSafetyError(f"KV count excessivo: {kv_count}")
                     
                self.header = GGUFHeader(magic, version, tensor_count, kv_count)
                
                self._parse_kv_pairs(f, kv_count)
                self._validate_metadata_consistency()
                
        except struct.error as e:
            raise HIPSafetyError(f"Erro de struct lendo GGUF: {e}")
            
        logger.info("Validação do modelo concluída com sucesso.")
        return True
