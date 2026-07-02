"""
Descoberta automática da amdhip64.dll no Windows.
Escaneia caminhos padrão do AMD HIP SDK sem exigir configuração manual.
"""

import os
import sys
from pathlib import Path
from typing import Optional

def find_hip_dll() -> Optional[str]:
    """
    Busca amdhip64.dll em caminhos padrão do Windows.
    
    Ordem de busca:
    1. Variável de ambiente HIP_PATH
    2. C:\\Program Files\\AMD\\HIP\\bin\\
    3. C:\\Program Files (x86)\\AMD\\HIP\\bin\\
    4. PATH do sistema
    
    Returns:
        Path completo para amdhip64.dll ou None se não encontrado
    """
    if sys.platform != "win32":
        return None
    
    hip_path = os.environ.get("HIP_PATH")
    dll_names = ["amdhip64.dll", "amdhip64_6.dll"]
    
    if hip_path:
        for name in dll_names:
            dll_path = Path(hip_path) / "bin" / name
            if dll_path.exists():
                return str(dll_path)
    
    standard_paths = [
        r"C:\Program Files\AMD\HIP\bin",
        r"C:\Program Files (x86)\AMD\HIP\bin",
    ]
    
    for path in standard_paths:
        for name in dll_names:
            candidate = Path(path) / name
            if candidate.exists():
                return str(candidate)
    
    rocm_base = Path(r"C:\Program Files\AMD\ROCm")
    if rocm_base.exists():

        for version_dir in rocm_base.iterdir():
            if version_dir.is_dir():
                for name in dll_names:
                    dll_candidate = version_dir / "bin" / name
                    if dll_candidate.exists():
                        return str(dll_candidate)
                    
    path_dirs = os.environ.get("PATH", "").split(";")
    for dir_path in path_dirs:
        for name in dll_names:
            dll_candidate = Path(dir_path) / name
            if dll_candidate.exists():
                return str(dll_candidate)
    
    return None

def validate_hip_installation() -> tuple[bool, str]:
    """
    Valida que o AMD HIP SDK está instalado corretamente.
    
    Returns:
        (success, message)
    """
    dll_path = find_hip_dll()
    
    if dll_path is None:
        return False, (
            "AMD HIP SDK não encontrado!\n\n"
            "Instale o HIP SDK:\n"
            "1. Baixe em: https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html\n"
            "2. Instale e reinicie o terminal\n"
            "3. Verifique que a variável HIP_PATH está definida"
        )
    
    import subprocess
    import shutil
    
    if not shutil.which("hipcc") or not shutil.which("hipcc.bat"):
        bin_dir = Path(dll_path).parent
        hipcc_bat = bin_dir / "hipcc.bat"
        hipcc_exe = bin_dir / "hipcc.exe"
        if hipcc_bat.exists() or hipcc_exe.exists():
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
            
    try:
        result = subprocess.run(
            ["hipcc", "--version"],
            capture_output=True,
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        if result.returncode != 0:
            return False, "hipcc encontrado, mas retornou erro ao checar versão."
    except FileNotFoundError:
        return False, f"hipcc não encontrado. A DLL está em {dll_path}, mas o compilador está ausente."
    
    return True, f"HIP SDK encontrado: {dll_path}"
