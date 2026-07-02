import flet as ft
from typing import Callable

class ChatView(ft.Container):
    def __init__(self, on_submit: Callable[[str], None], on_cancel: Callable[[], None]):
        super().__init__()
        self.expand = True
        self.on_submit = on_submit
        self.on_cancel = on_cancel
        
        self.chat_list = ft.ListView(expand=True, spacing=10, auto_scroll=True)
        self.input_field = ft.TextField(
            hint_text="Digite o prompt para o Qwen2.5...",
            expand=True,
            border_color="#333333",
            cursor_color="#ED1C24",
            on_submit=self._handle_submit
        )
        self.send_btn = ft.IconButton(
            icon=ft.icons.SEND,
            icon_color="#ED1C24",
            on_click=self._handle_submit
        )
        self.cancel_btn = ft.IconButton(
            icon=ft.icons.STOP_CIRCLE,
            icon_color="#ED1C24",
            visible=False,
            on_click=self._handle_cancel
        )
        
        self.content = ft.Column([
            self.chat_list,
            ft.Row([
                self.input_field,
                self.send_btn,
                self.cancel_btn
            ])
        ])
        
        self.current_reply = None
        
    def _handle_submit(self, e):
        if not self.input_field.value:
            return
            
        prompt = self.input_field.value
        self.input_field.value = ""
        self.input_field.disabled = True
        self.send_btn.visible = False
        self.cancel_btn.visible = True
        self.update()
        
        self.chat_list.controls.append(
            ft.Row([
                ft.Container(
                    content=ft.Text(prompt, color="#FFFFFF"),
                    padding=10,
                    bgcolor="#2A2A2D",
                    border_radius=10
                )
            ], alignment=ft.MainAxisAlignment.END)
        )
        
        self.current_reply = ft.Text("", color="#00FF41")
        self.chat_list.controls.append(
            ft.Row([
                ft.Container(
                    content=self.current_reply,
                    padding=10,
                    bgcolor="#1A1A1D",
                    border_radius=10,
                    border=ft.border.all(1, "#333333")
                )
            ], alignment=ft.MainAxisAlignment.START)
        )
        self.update()
        self.on_submit(prompt)

    def _handle_cancel(self, e):
        self.on_cancel()
        
    def append_token(self, token: str):
        if self.current_reply:
            self.current_reply.value += token
            self.update()
            
    def end_generation(self):
        self.input_field.disabled = False
        self.send_btn.visible = True
        self.cancel_btn.visible = False
        self.current_reply = None
        self.update()
