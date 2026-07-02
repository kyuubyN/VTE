import time
import threading
from multiprocessing.connection import Connection
import os
from vte.core.ipc import (
    UIMsgPrompt, UIMsgCancel, UIMsgShutdown,
    MotorMsgToken, MotorMsgMetrics, MotorMsgProgress, MotorMsgReady, MotorMsgError,
    MotorMsgStatusUpdate
)
from vte.core.model import VTEModel
from vte.core.gpu_monitor import GPUMonitor
from vte.bridge.logger import get_logger

logger = get_logger("VTE.Motor")

class InferenceEngine:
    def __init__(self, pipe_conn: Connection):
        self.conn = pipe_conn
        self.running = True
        self.is_generating = False
        
        self.token_buffer = ""
        self.last_flush_time = 0.0
        self.flush_interval = 0.050
        
        self.model_name = "qwen2.5:1.5b-q4_k_m"
        self.model = None
        
        self.telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True)

    def boot(self):
        """Inicializa as Fases 0 a 3 através do VTEModel"""
        self.conn.send(MotorMsgProgress("Iniciando Model Lifecycle...", 10))
        
        try:
            self.model = VTEModel.from_pretrained(
                self.model_name,
                idle_timeout_seconds=300,
                enable_auto_unload=True
            )
            self.conn.send(MotorMsgProgress("Modelo carregado na VRAM...", 100))
        except Exception as e:
            self.conn.send(MotorMsgError(f"Falha no boot: {e}"))
            self.running = False
            return
            
        self.conn.send(MotorMsgReady())
        self.telemetry_thread.start()

    def _telemetry_loop(self):
        """Thread isolada para nao travar o loop de inferencia ou o IPC"""
        monitor = None
        
        while self.running:
            if not monitor and self.model and self.model._hip:
                monitor = GPUMonitor(self.model._hip)
                
            if monitor:
                metrics = monitor.get_gpu_metrics()
                temp = metrics.get('temperature', 0.0)
                clock = 0.0
                vram = metrics.get('vram_allocated_mb', 0.0)
                total_vram = metrics.get('vram_total_system_mb', 8192.0)
                vram_free = total_vram - vram
                power = 0.0
            else:
                temp = 65.0
                clock = 2200.0
                vram = 1033.0
                vram_free = 8192.0 - 1033.0
                power = 120.0
                
            tps = 85.0 if self.is_generating else 0.0
            
            vram_details = None
            if self.model and self.model._is_loaded:
                try:
                    vram_details = self.model.get_vram_usage()
                    if vram_details and vram_details.get('total_mb', 0) > 0:
                        vram = vram_details['total_mb']
                except Exception:
                    pass
                    
            try:
                self.conn.send(MotorMsgMetrics(temp, clock, vram, power, tps, vram_free_system_mb=vram_free, vram_details=vram_details))
                
                if self.model:
                    status = self.model.get_model_status()
                    self.conn.send(MotorMsgStatusUpdate(
                        status["is_loaded"],
                        status["time_until_unload"]
                    ))
            except Exception:
                break
                
            time.sleep(1.0)

    def flush_tokens(self, force=False):
        """Dispara o buffer de tokens se o limite de 50ms foi atingido"""
        if not self.token_buffer:
            return
            
        now = time.perf_counter()
        if force or (now - self.last_flush_time >= self.flush_interval):
            self.conn.send(MotorMsgToken(self.token_buffer))
            self.token_buffer = ""
            self.last_flush_time = now

    def generate(self, prompt: str, max_tokens: int):
        self.is_generating = True
        
        try:
            generator = self.model.generate(prompt, max_tokens=max_tokens)
            
            for word in generator:

                if self.conn.poll():
                    msg = self.conn.recv()
                    if isinstance(msg, UIMsgCancel):
                        generator.close()
                        break
                    elif isinstance(msg, UIMsgShutdown):
                        generator.close()
                        self.running = False
                        return
                
                self.token_buffer += word
                self.flush_tokens()
                
            self.flush_tokens(force=True)
            
        except Exception as e:
            self.conn.send(MotorMsgError(f"Engine Panic: {str(e)}"))
        finally:
            self.is_generating = False

    def loop(self):
        self.boot()
        while self.running:
            try:
                if self.conn.poll(0.1):
                    msg = self.conn.recv()
                    
                    if isinstance(msg, UIMsgPrompt):
                        self.generate(msg.text, msg.max_tokens)
                        
                    elif isinstance(msg, UIMsgShutdown):
                        self.running = False
                        break
            except EOFError:
                break
            except KeyboardInterrupt:
                break
            except Exception as e:
                # Isolamento de Falhas: Ignorar pacotes malformados sem desligar o motor
                logger.warning(f"Pacote IPC ignorado devido a erro de parsing: {e}")

def motor_entry(pipe_conn: Connection):
    """Entry point do subprocesso"""
    engine = InferenceEngine(pipe_conn)
    engine.loop()
