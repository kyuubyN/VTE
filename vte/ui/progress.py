import flet as ft
from vte.core.ipc import MotorMsgProgress

class CompilationProgress(ft.Container):
    def __init__(self):
        super().__init__()
        self.expand = True
        self.bgcolor = "#0E0E10"
        self.alignment = ft.alignment.center
        
        self.title = ft.Text("Inicializando Motor VTE", size=24, color="#E0E0E0", weight=ft.FontWeight.BOLD)
        self.status = ft.Text("Preparando HIP Runtime...", size=14, color="#888888")
        self.progress_bar = ft.ProgressBar(width=400, color="#ED1C24", bgcolor="#333333")
        
        self.content = ft.Column([
            self.title,
            self.progress_bar,
            self.status
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER, alignment=ft.MainAxisAlignment.CENTER)

    def update_progress(self, msg: MotorMsgProgress):
        self.status.value = msg.status
        self.progress_bar.value = msg.percentage / 100.0
        self.update()
