import multiprocessing
import sys
import signal
import flet as ft
from vte.core.motor import motor_entry
from vte.ui.app import VTEApp
from vte.core.ipc import UIMsgShutdown

class Orchestrator:
    def __init__(self):
        self.motor_process = None
        self.pipe_parent = None
        self.pipe_child = None
        self.app_ref = None
        
    def start_motor(self):
        """Mata o motor antigo se existir e cria um novo"""
        if self.motor_process and self.motor_process.is_alive():
            try:
                self.pipe_parent.send(UIMsgShutdown())
                self.motor_process.join(timeout=2)
            except Exception:
                pass
            if self.motor_process.is_alive():
                self.motor_process.terminate()
                
        self.pipe_parent, self.pipe_child = multiprocessing.Pipe()
        self.motor_process = multiprocessing.Process(
            target=motor_entry, 
            args=(self.pipe_child,),
            daemon=True
        )
        self.motor_process.start()
        
        if self.app_ref:
            self.app_ref.pipe_conn = self.pipe_parent

            self.app_ref.main_layout.visible = False
            self.app_ref.progress_screen.visible = True
            self.app_ref.page.update()

    def main_flet(self, page: ft.Page):

        def on_window_event(e):
            if e.data == "close":
                self.pipe_parent.send(UIMsgShutdown())
                page.window_destroy()
                
        page.window_prevent_close = True
        page.on_window_event = on_window_event
        
        self.app_ref = VTEApp(page, self.pipe_parent, restart_callback=self.start_motor)

def cli_main():

    multiprocessing.freeze_support()
    
    orchestrator = Orchestrator()
    orchestrator.start_motor()
    
    def cleanup_handler(signum, frame):
        if orchestrator.pipe_parent:
            try:
                orchestrator.pipe_parent.send(UIMsgShutdown())
            except Exception:
                pass
        if orchestrator.motor_process:
            orchestrator.motor_process.join(timeout=5)
        sys.exit(0)
        
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    
    ft.app(target=orchestrator.main_flet, view=ft.AppView.WINDOW)

if __name__ == '__main__':
    cli_main()
