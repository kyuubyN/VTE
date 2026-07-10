"""
Descoberta automática da amdhip64.dll no Windows.
Escaneia caminhos padrão do AMD HIP SDK sem exigir configuração manual.
"""

import os
import sys
from pathlib import Path
from typing import Optional


def _find_amdhip64_in(directory: Path) -> Optional[Path]:
    """Acha a DLL do HIP runtime num diretório, sem assumir um sufixo de
    versão fixo: o instalador nomeia o arquivo `amdhip64_<major>.dll` (ex.
    `amdhip64_6.dll` no ROCm 6.x, `amdhip64_7.dll` no 7.x, confirmado real
    ao instalar o ROCm 7.1: só existe `amdhip64_7.dll`, sem alias sem
    versão) -- travar numa lista fixa de nomes quebra a cada bump de major
    version. Prefere `amdhip64.dll` (alias sem versão, quando existe) e
    cai para o primeiro `amdhip64_*.dll` encontrado, ordenado do mais novo
    pro mais antigo (maior número de versão primeiro).
    """
    if not directory.is_dir():
        return None
    unversioned = directory / "amdhip64.dll"
    if unversioned.exists():
        return unversioned
    candidates = sorted(directory.glob("amdhip64_*.dll"), reverse=True)
    return candidates[0] if candidates else None


def find_hip_dll() -> Optional[str]:
    """
    Busca amdhip64.dll (ou amdhip64_<versao>.dll) em caminhos padrão do Windows.

    Ordem de busca:
    1. Variável de ambiente HIP_PATH
    2. C:\\Program Files\\AMD\\HIP\\bin\\
    3. C:\\Program Files (x86)\\AMD\\HIP\\bin\\
    4. C:\\Program Files\\AMD\\ROCm\\<versao>\\bin\\ (versão mais nova primeiro)
    5. PATH do sistema

    Returns:
        Path completo para a DLL do HIP runtime ou None se não encontrado
    """
    if sys.platform != "win32":
        return None

    hip_path = os.environ.get("HIP_PATH")
    if hip_path:
        found = _find_amdhip64_in(Path(hip_path) / "bin")
        if found:
            return str(found)

    standard_paths = [
        r"C:\Program Files\AMD\HIP\bin",
        r"C:\Program Files (x86)\AMD\HIP\bin",
    ]

    for path in standard_paths:
        found = _find_amdhip64_in(Path(path))
        if found:
            return str(found)

    rocm_base = Path(r"C:\Program Files\AMD\ROCm")
    if rocm_base.exists():
        # Versões mais novas primeiro (ex. 7.1 antes de 6.4), caso mais de
        # uma esteja instalada lado a lado.
        version_dirs = sorted(
            (d for d in rocm_base.iterdir() if d.is_dir()),
            key=lambda d: d.name,
            reverse=True,
        )
        for version_dir in version_dirs:
            found = _find_amdhip64_in(version_dir / "bin")
            if found:
                return str(found)

    path_dirs = os.environ.get("PATH", "").split(";")
    for dir_path in path_dirs:
        found = _find_amdhip64_in(Path(dir_path))
        if found:
            return str(found)

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
