import re
import os
import hashlib
import subprocess
import shutil
from pathlib import Path
from vte.bridge.errors import HIPSafetyError
from vte.bridge.logger import get_logger

logger = get_logger(__name__)

def _sanitize_var(value: any) -> str:
    """Garante que apenas números e strings alfanuméricas entrem no template C++"""
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):

        safe_str = re.sub(r'[^a-zA-Z0-9_]', '', value)
        if safe_str != value:
            raise HIPSafetyError(f"Variável de template insegura bloqueada: {value}")
        return safe_str
    raise HIPSafetyError(f"Tipo de variável não suportado no Codegen: {type(value)}")

def _sanitize_mega_kernel_params(params: dict) -> dict:
    """Sanitização relaxada para mega-kernels (controlada)"""
    ALLOWED_TYPES = (int, float, str, list, tuple, dict)
    ALLOWED_LIST_TYPES = (int, float)
    sanitized = {}
    
    for key, value in params.items():
        if isinstance(value, ALLOWED_TYPES):
            if isinstance(value, (list, tuple)):
                if not all(isinstance(x, ALLOWED_LIST_TYPES) for x in value):
                    raise HIPSafetyError(f"Lista inválida em mega-kernel param {key}: {value}")
                sanitized[key] = value
            elif isinstance(value, str):
                safe_str = re.sub(r'[^a-zA-Z0-9_]', '', value)
                if safe_str != value:
                    raise HIPSafetyError(f"String insegura em mega-kernel param {key}: {value}")
                sanitized[key] = safe_str
            else:
                sanitized[key] = value
        else:
            raise HIPSafetyError(f"Tipo não permitido em mega-kernel: {type(value)}")
    return sanitized

def get_kernel_cache_path(cpp_code: str, arch: str) -> str:
    """Retorna o caminho do cache (.hsaco) baseado no hash do código renderizado"""
    code_hash = hashlib.sha256(cpp_code.encode('utf-8')).hexdigest()[:16]
    return f"cache/kernels/{arch}/kernel_{code_hash}.hsaco"

class CodegenEngine:
    def __init__(self):
        self.templates_dir = Path(__file__).parent / "templates"
        self.logs_dir = Path("logs/Codegen")
        self.cache_dir = Path("cache/kernels")
        
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def render_template(self, template_name: str, **kwargs) -> str:
        """Lê um template e substitui as variáveis seguras."""
        template_path = self.templates_dir / f"{template_name}.hip.template"
        
        if not template_path.exists():

            logger.warning(f"Template não encontrado no disco: {template_path}")
            return ""
            
        with open(template_path, "r", encoding="utf-8") as f:
            template_code = f.read()
            
        is_mega_kernel = kwargs.pop("is_mega_kernel", False)
        
        rendered_code = template_code
        if is_mega_kernel:
            safe_kwargs = _sanitize_mega_kernel_params(kwargs)
            for k, v in safe_kwargs.items():

                if isinstance(v, (list, tuple)):
                    v = "{" + ", ".join(map(str, v)) + "}"
                rendered_code = rendered_code.replace(f"{{{{ {k} }}}}", str(v))
        else:
            for k, v in kwargs.items():
                safe_val = _sanitize_var(v)
                rendered_code = rendered_code.replace(f"{{{{ {k} }}}}", safe_val)
                
        return rendered_code
        
    def _parse_vgpr_usage(self, compiler_output: str) -> int:
        """Extrai VGPR usage do output do hipcc para detectar Register Spilling"""

        match = re.search(r'vgpr_count:\s*(\d+)', compiler_output, re.IGNORECASE)
        if not match:
            match = re.search(r'VGPRs:\s*(\d+)', compiler_output, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return -1
        
    def _setup_hip_env(self) -> dict:
        """Prepara variáveis de ambiente para o hipcc funcionar no Windows com MSVC."""
        import glob
        env = os.environ.copy()
        
        rocm_bin = r"C:\Program Files\AMD\ROCm\6.4\bin"
        if rocm_bin not in env.get("PATH", ""):
            env["PATH"] = rocm_bin + os.pathsep + env.get("PATH", "")
        
        vs_roots = glob.glob(r"C:\Program Files (x86)\Microsoft Visual Studio\*\BuildTools\VC\Tools\MSVC\*\include")
        ucrt_roots = glob.glob(r"C:\Program Files (x86)\Windows Kits\10\Include\*\ucrt")
        um_roots   = glob.glob(r"C:\Program Files (x86)\Windows Kits\10\Include\*\um")
        shared_roots = glob.glob(r"C:\Program Files (x86)\Windows Kits\10\Include\*\shared")
        
        extra_includes = []
        if vs_roots:     extra_includes.append(sorted(vs_roots)[-1])
        if ucrt_roots:   extra_includes.append(sorted(ucrt_roots)[-1])
        if um_roots:     extra_includes.append(sorted(um_roots)[-1])
        if shared_roots: extra_includes.append(sorted(shared_roots)[-1])
        
        if extra_includes:
            env["INCLUDE"] = os.pathsep.join(extra_includes) + os.pathsep + env.get("INCLUDE", "")
            logger.debug(f"MSVC/SDK includes injetados: {extra_includes}")
        else:
            logger.warning("Headers MSVC/SDK não encontrados. hipcc pode falhar com 'cmath not found'.")
        
        return env

    def compile_kernel(self, template_name: str, arch: str, **kwargs) -> str:
        """Renderiza e compila o kernel via hipcc"""
        cpp_code = self.render_template(template_name, **kwargs)
        if not cpp_code:
            cpp_code = kwargs.get("mock_code", "")
            if not cpp_code:
                 cpp_code = str(kwargs)
                 
        cache_path = get_kernel_cache_path(cpp_code, arch)
        
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        
        code_hash = hashlib.sha256(cpp_code.encode('utf-8')).hexdigest()[:16]
        log_file = self.logs_dir / f"{template_name}_{code_hash}.hip"
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(cpp_code)
            
        if Path(cache_path).exists():
            logger.debug(f"Cache hit: {cache_path}")
            return cache_path
        
        env = self._setup_hip_env()
            
        if not shutil.which("hipcc", path=env.get("PATH")):
            logger.warning(f"hipcc não encontrado no PATH. Simulando compilaão de {cache_path}")
            with open(cache_path, "wb") as f:
                f.write(b"MOCK_HSACO")
            return cache_path
            
        cmd = [
            "hipcc",
            "--genco",
            f"--offload-arch={arch}",
            "-O3",
            str(log_file),
            "-o",
            cache_path
        ]
        
        logger.info(f"Compilando kernel {template_name} para {arch}...")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
            logger.info(f"Kernel compilado: {cache_path}")
            
            vgpr_usage = self._parse_vgpr_usage(result.stderr + result.stdout)
            if vgpr_usage > 128:
                logger.error(f"Kernel {template_name} usa {vgpr_usage} VGPRs (limite: 128). Register spilling detectado.")
                raise HIPSafetyError(f"VGPR usage excedeu o limite seguro (128): {vgpr_usage}")
                
        except subprocess.CalledProcessError as e:
            raise HIPSafetyError(f"Falha na compilação do kernel {template_name}:\n{e.stderr}")
            
        return cache_path
