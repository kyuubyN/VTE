import struct
import hashlib
from pathlib import Path
from dataclasses import dataclass
from vte.bridge.errors import HIPSafetyError
from vte.bridge.logger import get_logger

logger = get_logger(__name__)

# Tabela de arquiteturas suportadas: cada entrada é o "perfil de validação"
# daquela arquitetura -- substitui os 3 gates que antes eram literais únicos
# hardcoded (architecture != "qwen2", block_count != 28, nome de arquivo
# exato). Arquitetura desconhecida continua sendo um gate DURO (rejeita),
# só que parametrizado por entrada da tabela em vez de comparação única.
# Números confirmados contra os GGUFs reais (gguf.GGUFReader), não
# documentação de terceiros -- ver plano em curious-roaming-quasar.md.
SUPPORTED_ARCHITECTURES = {
    "qwen2": {
        "block_count": 28,
        "max_context_length": 100_000,
        "expected_filename": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        # Nomes de atributos em vte/config.py (não valores literais): lidos
        # dinamicamente em validate() para permanecerem configuráveis/
        # monkeypatcháveis em um único lugar, mesmo padrão do hash abaixo.
        "size_min_key": "ALLOWED_MODEL_SIZE_MIN",
        "size_max_key": "ALLOWED_MODEL_SIZE_MAX",
        "hash_config_key": "QWEN2_5_EXPECTED_HASH",
    },
    "granite": {
        "block_count": 40,
        # Contexto nativo do Granite 4.1 é 131072 (bem maior que os 32768 do
        # Qwen2.5) -- o teto de sanidade precisa acomodar isso sem virar um
        # cheque em branco.
        "max_context_length": 200_000,
        "expected_filename": "granite-4.1-3b-Q8_0.gguf",
        "size_min_key": "GRANITE_4_1_3B_SIZE_MIN",
        "size_max_key": "GRANITE_4_1_3B_SIZE_MAX",
        "hash_config_key": "GRANITE_4_1_3B_EXPECTED_HASH",
    },
    "qwen35": {
        # 24 camadas no total (6 full_attention + 18 linear_attention/Gated
        # DeltaNet) -- block_count no GGUF conta TODAS, não só as de
        # atenção completa (confirmado via gguf.GGUFReader real).
        "block_count": 24,
        # Contexto nativo real é 262144 (config.json:
        # max_position_embeddings) -- maior que o do Granite, teto de
        # sanidade generoso o bastante sem virar cheque em branco.
        "max_context_length": 300_000,
        "expected_filename": "Qwen3.5-2B-Q6_K.gguf",
        "size_min_key": "QWEN35_2B_SIZE_MIN",
        "size_max_key": "QWEN35_2B_SIZE_MAX",
        "hash_config_key": "QWEN35_2B_EXPECTED_HASH",
    },
}


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

    def _validate_or_generate_hash(self, profile: dict) -> bool:
        """
        Se o hash esperado (config.py, chave `profile['hash_config_key']`)
        estiver vazio/placeholder, calcula o hash e exibe para o usuário
        configurar -- mesmo mecanismo de antes, agora parametrizado pela
        arquitetura detectada em vez de uma única constante do Qwen.
        """
        import vte.config as config

        expected_hash = getattr(config, profile["hash_config_key"], None)
        calculated_hash = self._calculate_sha256()

        if not expected_hash or expected_hash == "sha256:PLACEHOLDER":
            logger.warning(
                f"PRIMEIRA EXECUÇÃO - Hash não configurado. Adicione no config.py: "
                f"{profile['hash_config_key']} = '{calculated_hash}'"
            )
            return True

        if calculated_hash != expected_hash:
            raise HIPSafetyError(
                f"Hash não corresponde!\n"
                f"Calculado: {calculated_hash}\n"
                f"Esperado:  {expected_hash}"
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

            # Chaves de hiperparâmetro são lidas pelo SUFIXO (".context_length",
            # ".embedding_length", ".block_count") em vez de um prefixo
            # "qwen2." fixo -- funciona para qualquer arquitetura (qwen2.*,
            # granite.*, etc.) sem precisar saber a arquitetura de antemão
            # (a ordem dos pares chave/valor no GGUF não é garantida, então
            # não dá pra depender de "general.architecture" já ter sido lido
            # antes destas chaves). Verificado contra os dois GGUFs reais do
            # projeto: nenhuma outra chave colide com esses sufixos.
            if key == "general.architecture":
                if vtype == 8:
                    self.header.architecture = self._read_string(f)
                else:
                    raise HIPSafetyError("general.architecture não é string")
            elif key.endswith(".context_length"):
                if vtype == 4:
                    self.header.context_length = struct.unpack("<I", f.read(4))[0]
                else:
                    self._skip_value(f, vtype)
            elif key.endswith(".embedding_length"):
                if vtype == 4:
                     self.header.embedding_length = struct.unpack("<I", f.read(4))[0]
                else:
                     self._skip_value(f, vtype)
            elif key.endswith(".block_count"):
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

    def _validate_metadata_consistency(self) -> dict:
        """Valida que os metadados são consistentes com o PERFIL da arquitetura
        detectada (`SUPPORTED_ARCHITECTURES`) e retorna esse perfil, usado em
        seguida por `validate()` para checar nome de arquivo/tamanho/hash --
        checagens que antes eram feitas ANTES de saber a arquitetura (só
        funcionava porque só existia uma arquitetura possível)."""
        profile = SUPPORTED_ARCHITECTURES.get(self.header.architecture)
        if profile is None:
            supported = ", ".join(SUPPORTED_ARCHITECTURES.keys())
            raise HIPSafetyError(
                f"Arquitetura não suportada: '{self.header.architecture}'. "
                f"Suportadas: {supported}"
            )

        if self.header.context_length > profile["max_context_length"]:
            raise HIPSafetyError(f"context_length suspeito: {self.header.context_length}")

        if self.header.block_count != profile["block_count"]:
            raise HIPSafetyError(
                f"Block count inválido para '{self.header.architecture}'. "
                f"Esperado: {profile['block_count']}, Recebido: {self.header.block_count}"
            )

        return profile

    def validate(self) -> bool:
        logger.info(f"Validando modelo: {self.path}")

        if not self.path.exists() or not self.path.is_file():
            raise HIPSafetyError("Modelo não encontrado")

        # Whitelist rápida por nome de arquivo, ANTES de tentar fazer parse
        # do conteúdo -- rejeita arquivos com nome desconhecido/malformado
        # sem precisar interpretar bytes não confiáveis primeiro (mesma
        # postura de segurança do gate original de nome único, agora
        # parametrizada por todas as arquiteturas suportadas).
        known_filenames = {p["expected_filename"] for p in SUPPORTED_ARCHITECTURES.values()}
        if self.path.name not in known_filenames:
            raise HIPSafetyError("Nome do modelo incorreto")

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
                profile = self._validate_metadata_consistency()

        except struct.error as e:
            raise HIPSafetyError(f"Erro de struct lendo GGUF: {e}")

        # Checagens que dependem de já saber a arquitetura (nome de arquivo,
        # tamanho, hash) -- feitas DEPOIS do parse, usando o perfil resolvido
        # em `_validate_metadata_consistency`.
        if self.path.name != profile["expected_filename"]:
            raise HIPSafetyError("Nome do modelo incorreto")

        import vte.config as config
        size = self.path.stat().st_size
        size_min = getattr(config, profile["size_min_key"])
        size_max = getattr(config, profile["size_max_key"])
        if size < size_min or size > size_max:
             raise HIPSafetyError(f"Tamanho fora do esperado: {size} bytes")

        self._validate_or_generate_hash(profile)

        logger.info("Validação do modelo concluída com sucesso.")
        return True
