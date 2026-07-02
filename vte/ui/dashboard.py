import flet as ft
from vte.core.ipc import MotorMsgMetrics, MotorMsgStatusUpdate

class Dashboard(ft.Container):
    def __init__(self):
        super().__init__()
        self.bgcolor = "#1A1A1D"
        self.border_radius = 8
        self.border = ft.border.all(1, "#333333")
        self.padding = 20
        self.width = 300
        
        self.temp_text = ft.Text("0 °C", size=24, weight=ft.FontWeight.BOLD, color="#E0E0E0")
        self.clock_text = ft.Text("0 MHz", size=24, weight=ft.FontWeight.BOLD, color="#E0E0E0")
        self.vram_text = ft.Text("0 MB", size=24, weight=ft.FontWeight.BOLD, color="#E0E0E0")
        
        # Sub-métricas de VRAM
        self.vram_weights_text = ft.Text("Weights: 0 MB", size=12, color="#888888")
        self.vram_kv_text = ft.Text("KV Cache: 0 MB", size=12, color="#888888")
        self.vram_arena_text = ft.Text("Arena: 0 MB", size=12, color="#888888")
        self.vram_free_text = ft.Text("Livre para o Sistema: 0 GB", size=12, color="#00FF41", weight=ft.FontWeight.BOLD)
        self.vram_details_col = ft.Column([
            self.vram_weights_text,
            self.vram_kv_text,
            self.vram_arena_text,
            ft.Container(height=4),
            self.vram_free_text
        ], spacing=0, visible=False)
        self.tps_text = ft.Text("0.0 t/s", size=24, weight=ft.FontWeight.BOLD, color="#00FF41")
        self.status_text = ft.Text("Inicializando...", size=14, color="#AAAAAA")
        
        self.content = ft.Column(
            controls=[
                ft.Text("AMD RDNA3 Telemetry", size=16, weight=ft.FontWeight.BOLD, color="#ED1C24"),
                ft.Divider(color="#333333"),
                self._build_metric("Temperature", self.temp_text),
                self._build_metric("Core Clock", self.clock_text),
                ft.Column([
                    ft.Text("VRAM Usage", size=12, color="#888888"),
                    self.vram_text,
                    self.vram_details_col
                ], spacing=2),
                ft.Divider(color="#333333"),
                self._build_metric("Inference Speed", self.tps_text),
                ft.Divider(color="#333333"),
                self._build_metric("Model Lifecycle", self.status_text),
            ],
            spacing=10
        )
        
    def _build_metric(self, label: str, value_control: ft.Control):
        return ft.Column([
            ft.Text(label, size=12, color="#888888"),
            value_control
        ], spacing=2)

    def update_metrics(self, msg: MotorMsgMetrics):
        self.temp_text.value = f"{msg.temp_c:.1f} °C"
        
        if msg.temp_c > 85.0:
            self.temp_text.color = "#ED1C24"
        else:
            self.temp_text.color = "#E0E0E0"
            
        self.clock_text.value = f"{msg.clock_mhz:.0f} MHz"
        self.vram_text.value = f"{msg.vram_mb:.0f} MB"
        
        if msg.vram_details and msg.vram_details.get('total_mb', 0) > 0:
            self.vram_weights_text.value = f"Weights: {msg.vram_details.get('weights_mb', 0):.0f} MB"
            self.vram_kv_text.value = f"KV Cache: {msg.vram_details.get('kv_cache_mb', 0):.0f} MB"
            self.vram_arena_text.value = f"Arena: {msg.vram_details.get('activations_mb', 0):.0f} MB"
            self.vram_free_text.value = f"Livre para o Sistema: {msg.vram_free_system_mb / 1024:.1f} GB"
            self.vram_details_col.visible = True
        else:
            self.vram_details_col.visible = False
            
        self.tps_text.value = f"{msg.tokens_sec:.1f} t/s"
        self.update()

    def update_lifecycle_status(self, msg: MotorMsgStatusUpdate):
        if not msg.is_loaded:
            self.status_text.value = " Modelo descarregado (ocioso)"
            self.status_text.color = "#AAAAAA"
        else:
            if msg.time_until_unload is not None:
                self.status_text.value = f" Ativo (unload em {msg.time_until_unload:.0f}s)"
            else:
                self.status_text.value = " Ativo"
            self.status_text.color = "#00FF41"
        self.update()
