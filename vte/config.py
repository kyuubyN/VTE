import sys
import ctypes.util
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "Model"
MODEL_PATH = MODEL_DIR / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
LOG_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"
KERNELS_DIR = PROJECT_ROOT / "kernels"

MAX_ALLOCATION_SIZE = 8 * 1024 * 1024 * 1024
VRAM_SAFETY_MARGIN = 0.95
MIN_VRAM_REQUIRED = 4 * 1024 * 1024 * 1024
TDR_TIMEOUT_MS = 2000
# Teto de tempo (ms) para uma sincronização de GPU (hipDeviceSynchronize) antes
# do KernelWatchdog considerar a GPU travada e entrar em PANIC MODE, bloqueando
# novos lançamentos. 8s cobre um forward pass completo de 28 camadas mesmo sem
# cache de kernel "quente", mas ainda protege contra loops/kernels realmente
# travados (limite de "processador" da GPU, complementar ao VRAM_SAFETY_MARGIN).
SAFE_DISPATCH_TIMEOUT = 8000
CACHE_LINE_SIZE = 64

ALLOWED_MODEL_HASH = "sha256:PLACEHOLDER"
QWEN2_5_EXPECTED_HASH = "sha256:PLACEHOLDER"
GRANITE_4_1_3B_EXPECTED_HASH = "sha256:PLACEHOLDER"
QWEN35_2B_EXPECTED_HASH = "sha256:PLACEHOLDER"
ALLOWED_MODEL_SIZE_MIN = 800 * 1024 * 1024
ALLOWED_MODEL_SIZE_MAX = 1200 * 1024 * 1024
GRANITE_4_1_3B_SIZE_MIN = 3 * 1024 * 1024 * 1024
GRANITE_4_1_3B_SIZE_MAX = 4 * 1024 * 1024 * 1024
# Arquivo real medido nesta sessão: 1574961408 bytes (~1.47GB) -- margem
# generosa em torno disso, mesmo padrão de banda usado pros outros dois
# modelos (não um cheque em branco, mas não colado no byte exato).
QWEN35_2B_SIZE_MIN = 1 * 1024 * 1024 * 1024
QWEN35_2B_SIZE_MAX = 2 * 1024 * 1024 * 1024

DEFAULT_GPU_ARCH = "gfx1100"
GPU_ARCH_MAP = {
    "rx 7600": "gfx1102", "rx 7600 xt": "gfx1101", "rx 7700": "gfx1101",
    "rx 7700 xt": "gfx1101", "rx 7800 xt": "gfx1101", "rx 7900": "gfx1100",
    "rx 7900 xt": "gfx1100", "rx 7900 xtx": "gfx1100",
    "rx 6600": "gfx1032", "rx 6600 xt": "gfx1032", "rx 6700": "gfx1031",
    "rx 6700 xt": "gfx1031", "rx 6800": "gfx1030", "rx 6800 xt": "gfx1030",
    "rx 6900": "gfx1030", "rx 6900 xt": "gfx1030",
}

VRAM_USAGE_LIMIT = VRAM_SAFETY_MARGIN
MAX_SHARED_MEM_PER_BLOCK = 64 * 1024
MAX_GRID_DIMENSIONS = (2**31) - 1

def find_hip_dll() -> str | None:
    hip_path = os.environ.get("HIP_PATH")
    if hip_path:
        p = Path(hip_path) / "bin" / "amdhip64.dll"
        if p.exists(): return str(p)
    base = Path(r"C:\Program Files\AMD\ROCm")
    for version in ["6.4", "6.3", "6.2", "6.1", "6.0", "5.7"]:
        p = base / version / "bin" / "amdhip64.dll"
        if p.exists(): return str(p)
    p = Path(r"C:\Program Files\AMD\HIPinmdhip64.dll")
    if p.exists(): return str(p)
    lib = ctypes.util.find_library("amdhip64")
    if lib: return lib
    return None

def preflight_safety_check() -> tuple[bool, str]:
    if sys.platform != "win32":
        return False, "VTE requer Windows (WDDM resiliência)."
    dll = find_hip_dll()
    if not dll:
        return False, "amdhip64.dll não encontrada. Instale o ROCm/HIP SDK."
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True)
        return False, f"Diretório Model/ criado. Coloque o modelo em {MODEL_PATH}."
    if not MODEL_PATH.exists():
        return False, f"Modelo não encontrado em {MODEL_PATH}."
    size = MODEL_PATH.stat().st_size
    if size < ALLOWED_MODEL_SIZE_MIN or size > ALLOWED_MODEL_SIZE_MAX:
        return False, f"Tamanho do modelo fora do range permitido: {size} bytes."
    return True, "Preflight Ok."
