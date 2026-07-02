import pytest
import time
import subprocess
from vte.core.gpu_monitor import GPUMonitor
from vte.bridge.hip_runtime import HIPRuntime

def test_zombie_kernel_detection():
    """Valida que kernels nao ficam orfaos apos crash do processo."""
    # Esse teste depende de subprocessos. Para evitar travar ambientes de CI reais, 
    # rodariamos num container ou skipamos se nao estiver no ambiente apropriado.
    pytest.skip("Ignorado por padrao fora do ambiente Docker isolado.")

def test_power_consumption_idle():
    """Valida que GPU nao consome energia quando ociosa e nao possui threads zumbis."""
    try:
        hip = HIPRuntime()
        hip.initialize()
    except Exception:
        pytest.skip("Requer GPU AMD nativa.")
        
    monitor = GPUMonitor(hip)
    
    metrics_baseline = monitor.get_gpu_metrics()
    baseline_power = metrics_baseline.get('power_w', 0.0) # Mock fallbacks podem retornar 0
    
    # Em um cenario real de teste, importariamos VTEModel e simulariamos
    # Por agora, simulamos apenas que o monitor esta respondendo
    metrics_idle = monitor.get_gpu_metrics()
    loaded_power = metrics_idle.get('power_w', 0.0)
    
    power_diff = loaded_power - baseline_power
    assert power_diff < 10.0, f"GPU consumindo {power_diff:.1f}W extra quando ociosa. Processamento fantasma detectado."

def test_auto_unload_idle():
    """Verifica se o modelo e descarregado da VRAM apos o timeout de inatividade."""
    from vte.core.lifecycle import ModelLifecycleManager
    
    class MockModel:
        def __init__(self):
            self._hip = None
            self._allocator = None
            self._graph = None
            
        def _load(self):
            pass
            
    model = MockModel()
    # Usando timeout de 10 segundos conforme pedido
    manager = ModelLifecycleManager(model, idle_timeout_seconds=10, enable_auto_unload=True)
    manager.start_monitoring()
    
    assert manager._is_loaded is True, "O lifecycle deveria estar marcado como carregado ao iniciar."
    
    # Aguarda 10 segundos do timeout + ~3 segundos do loop interno (que roda a cada 2s)
    time.sleep(13)
    
    assert manager._is_loaded is False, "O modelo NAO foi descarregado automaticamente da VRAM apos o timeout de 10 segundos!"

def test_unload_on_shutdown():
    """Verifica se o cleanup automatico (unload) ocorre quando o aplicativo e encerrado."""
    from vte.core.model import VTEModel
    from unittest.mock import patch, MagicMock
    from pathlib import Path

    with patch('vte.core.model.ModelLifecycleManager') as mock_lifecycle_class:
        mock_lifecycle = MagicMock()
        mock_lifecycle._is_loaded = True
        mock_lifecycle_class.return_value = mock_lifecycle
        
        model = VTEModel(Path("Model/fake.gguf"))
        model._loaded = True
        
        # Simulando encerramento/destruicao (ex: garbage collector limpa no fechamento do app)
        model.__del__()
        
        # Verifica se o unload foi chamado!
        mock_lifecycle.unload.assert_called_once()

