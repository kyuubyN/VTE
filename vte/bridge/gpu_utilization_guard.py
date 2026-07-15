import os
import subprocess
import threading
import time
from vte.bridge.logger import get_logger

logger = get_logger(__name__)

_POWERSHELL_CMD_TEMPLATE = (
    "(Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction SilentlyContinue)."
    "CounterSamples | Where-Object {{ $_.Path -like '*pid_{pid}_*' -and $_.CookedValue -gt 0 }} | "
    "Measure-Object -Property CookedValue -Sum | Select-Object -ExpandProperty Sum"
)


class GPUUtilizationGuard:
    """
    Monitora (só observa, não interrompe) a utilização de GPU do processo
    atual via contadores de performance do Windows — um segundo sinal
    independente, além da própria medição interna de tempo do
    HIPRuntime._enforce_duty_cycle_limit, útil para diagnóstico (ex.: detectar
    se ALGO ALÉM do nosso próprio código está pressionando a GPU).

    IMPORTANTE: este monitor NUNCA interrompe/pausa a execução — quem
    regula de fato a GPU para não passar de ~95% é o limitador de duty cycle
    em HIPRuntime (_throttle_before_dispatch / _throttle_duty_cycle), que
    insere pequenas pausas ANTES de cada lançamento sem nunca lançar exceção.
    Um guard que interrompe o programa ao detectar uso alto seria o oposto do
    que se quer aqui: o objetivo é a GPU rodar continuamente regulada a 95%,
    nunca "parar" por ter chegado perto do teto.

    Só funciona no Windows (usa `Get-Counter`/WMI via PowerShell). Em qualquer
    falha (powershell ausente, contador indisponível, timeout) apenas loga um
    aviso e continua sem travar a aplicação — é uma ferramenta de
    observabilidade best-effort, não uma dependência crítica.
    """

    def __init__(
        self,
        watchdog,
        threshold_percent: float = 95.0,
        poll_interval_seconds: float = 1.0,
        sustained_samples: int = 3,
    ):
        self._watchdog = watchdog
        self._threshold = threshold_percent
        self._poll_interval = poll_interval_seconds
        self._sustained_samples = sustained_samples
        self._pid = os.getpid()

        self._running = False
        self._thread: threading.Thread | None = None
        self._consecutive_over = 0
        self._last_reading: float | None = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="GPUUtilizationGuard")
        self._thread.start()
        logger.info(f"GPUUtilizationGuard iniciado (limite {self._threshold}%, PID {self._pid}).")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("GPUUtilizationGuard parado.")

    def get_last_reading(self) -> float | None:
        return self._last_reading

    def _query_gpu_utilization(self) -> float | None:
        cmd = _POWERSHELL_CMD_TEMPLATE.format(pid=self._pid)
        try:
            # CREATE_NO_WINDOW é ESSENCIAL aqui: este `subprocess.run` roda
            # em loop contínuo (`_monitor_loop`, a cada `_poll_interval`).
            # Quando o processo pai é uma app GUI sem console (ex.: o backend
            # do Aetheris lançado via pythonw.exe), CADA chamada de
            # powershell abre uma NOVA janela de console visível -- o
            # sintoma "abre terminais do PowerShell infinitamente" relatado.
            # Com console anexado (python.exe normal) as janelas não
            # aparecem, por isso só se manifesta no app empacotado. Mesmo
            # padrão já usado em hip_runtime.py para o hipcc.
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            result = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                capture_output=True, text=True, timeout=5, creationflags=creationflags,
            )
            output = result.stdout.strip()
            if not output:
                return 0.0
            # PowerShell formata números conforme a cultura do sistema (ex.: pt-BR
            # usa vírgula decimal: "0,0082"). Normaliza para o formato que
            # float() do Python entende, independente do locale da máquina.
            return float(output.replace(",", "."))
        except Exception as e:
            logger.debug(f"GPUUtilizationGuard: falha ao consultar contador de GPU: {e}")
            return None

    def _monitor_loop(self):
        while self._running:
            time.sleep(self._poll_interval)
            if not self._running:
                break

            usage = self._query_gpu_utilization()
            if usage is None:
                continue

            self._last_reading = usage

            if usage >= self._threshold:
                self._consecutive_over += 1
                logger.debug(
                    f"Utilização de GPU (contador do Windows) em {usage:.1f}% "
                    f"(amostra {self._consecutive_over}/{self._sustained_samples} acima de {self._threshold}%). "
                    f"Apenas observação — quem regula é o duty cycle limiter em HIPRuntime."
                )
            else:
                self._consecutive_over = 0
