import flet as ft
import threading
from multiprocessing.connection import Connection
from vte.ui.chat_view import ChatView
from vte.ui.dashboard import Dashboard
from vte.ui.progress import CompilationProgress
from vte.core.ipc import (
    UIMsgPrompt, UIMsgCancel,
    MotorMsgToken, MotorMsgMetrics, MotorMsgProgress, MotorMsgReady, MotorMsgError,
    MotorMsgStatusUpdate
)

class VTEApp:
    def __init__(self, page: ft.Page, pipe_conn: Connection, restart_callback=None):
        self.page = page
        self.pipe_conn = pipe_conn
        self.restart_callback = restart_callback
        
        self.page.title = "VTE - RDNA3 Inference Engine"
        self.page.bgcolor = "#0E0E10"
        self.page.padding = 0
        self.page.theme_mode = ft.ThemeMode.DARK
        
        self.chat_view = ChatView(on_submit=self._submit_prompt, on_cancel=self._cancel_prompt)
        self.dashboard = Dashboard()
        self.progress_screen = CompilationProgress()
        
        self.main_layout = ft.Row([
            ft.Container(self.chat_view, expand=True, padding=20),
            ft.Container(self.dashboard, padding=20)
        ], expand=True, visible=False)
        
        self.page.add(self.progress_screen)
        self.page.add(self.main_layout)
        
        self.tdr_banner = ft.Banner(
            bgcolor="#ED1C24",
            leading=ft.Icon(ft.icons.WARNING_AMBER_ROUNDED, color="#FFFFFF", size=40),
            content=ft.Text(
                "Núcleo da GPU Abortado (Proteção WDDM Ativada). O contexto do driver foi reiniciado.",
                color="#FFFFFF"
            ),
            actions=[
                ft.TextButton("Reiniciar Motor", style=ft.ButtonStyle(color="#FFFFFF"), on_click=self._restart_motor)
            ]
        )

        pass        

        self.running = True
        self.listener_thread = threading.Thread(target=self._listen_pipe, daemon=True)
        self.listener_thread.start()

    def _submit_prompt(self, text: str):
        self.pipe_conn.send(UIMsgPrompt(text=text))
        
    def _cancel_prompt(self):
        self.pipe_conn.send(UIMsgCancel())

    def _restart_motor(self, e):
        self.page.close(self.tdr_banner)
        self.page.update()
        if self.restart_callback:
            self.restart_callback()

    def _show_crash(self):
        self.chat_view.end_generation()
        self.page.open(self.tdr_banner)
        self.page.update()

    def _listen_pipe(self):
        while self.running:
            try:
                if self.pipe_conn.poll(0.1):
                    msg = self.pipe_conn.recv()
                    
                    if isinstance(msg, MotorMsgProgress):
                        self.progress_screen.update_progress(msg)
                    
                    elif isinstance(msg, MotorMsgReady):
                        self.progress_screen.visible = False
                        self.main_layout.visible = True
                        self.page.update()
                        
                    elif isinstance(msg, MotorMsgToken):
                        self.chat_view.append_token(msg.text)
                        
                    elif isinstance(msg, MotorMsgMetrics):
                        self.dashboard.update_metrics(msg)
                        
                    elif isinstance(msg, MotorMsgStatusUpdate):
                        self.dashboard.update_lifecycle_status(msg)
                        
                    elif isinstance(msg, MotorMsgError):
                        self._show_crash()
                        break
            except EOFError:

                self._show_crash()
                break
            except OSError:
                break
