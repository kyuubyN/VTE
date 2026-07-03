"""
app.py — UI desktop do VTE (Flet 0.85+), tudo composto num único módulo por
simplicidade de inicialização (`vte-ui` aponta para `cli_main` aqui).

Arquitetura de processo (inalterada da versão anterior): o motor de
inferência roda em um `multiprocessing.Process` separado (vte.core.motor),
falado por um `multiprocessing.Pipe` — isso isola qualquer TDR/crash da GPU
do processo da UI, que continua vivo para mostrar o banner de recuperação e
permitir reiniciar o motor sem fechar a janela.

Ponte thread-safe UI <-> motor: a thread que lê o pipe (`_listen_pipe`) roda
fora do event loop do Flet. Em vez de mutar controles diretamente dali
(não seguro nesta versão do Flet — `Control.update()` assume estar rodando
no event loop da sessão), cada mensagem recebida é publicada via
`page.pubsub.send_all()` e consumida por um handler `async def` inscrito no
próprio event loop: `PubSubHub` despacha handlers assíncronos com
`asyncio.run_coroutine_threadsafe`, a única forma documentada de cruzar
threads com segurança nesta arquitetura.
"""
import multiprocessing
import signal
import sys
import threading
from multiprocessing.connection import Connection
from pathlib import Path

import flet as ft

from vte.core.ipc import (
    UIMsgPrompt, UIMsgCancel, UIMsgShutdown,
    MotorMsgToken, MotorMsgMetrics, MotorMsgProgress, MotorMsgReady, MotorMsgError,
    MotorMsgStatusUpdate, MotorMsgLog, MotorMsgDone,
)
from vte.core.motor import motor_entry, DEFAULT_CONTEXT_LENGTH
from vte.ui.theme import get_palette, Palette

CONTEXT_LENGTH_OPTIONS = [512, 1024, 2048, 4096, 8192]
MAX_LOG_LINES = 500

# Caminho absoluto (não depende do cwd de onde `vte-ui` é invocado) --
# vte/ui/app.py -> raiz do repo é dois níveis acima.
ASSETS_DIR = str(Path(__file__).resolve().parent.parent.parent / "assets")


class Orchestrator:
    """Dono do ciclo de vida do processo do motor E do pipe de leitura.

    Importante: exatamente UMA thread lê `pipe_parent.recv()` durante toda a
    vida de uma conexão -- multiprocessing.Connection não é seguro para
    múltiplos leitores concorrentes (dois `.recv()` simultâneos entrelaçam
    bytes no meio de um frame pickle e corrompem o stream para ambos).
    Colocar essa thread aqui, em vez de dentro de VTEApp, garante isso mesmo
    que `main_flet` seja chamado mais de uma vez para o mesmo processo (ex.:
    uma sessão reconectando) -- cada `VTEApp` novo só passa a ser o
    destinatário das mensagens, nunca um leitor adicional do pipe.
    """

    def __init__(self):
        self.motor_process = None
        self.pipe_parent = None
        self.pipe_child = None
        self.app_ref: "VTEApp" = None
        self.context_length = DEFAULT_CONTEXT_LENGTH
        self._listener_generation = 0

    def start_motor(self, context_length: int = None):
        if context_length is not None:
            self.context_length = context_length

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
            args=(self.pipe_child, self.context_length),
            daemon=True
        )
        self.motor_process.start()

        if self.app_ref:
            self.app_ref.reset_for_new_motor(self.pipe_parent, self.context_length)

        self._listener_generation += 1
        generation = self._listener_generation
        threading.Thread(
            target=self._listen_pipe, args=(self.pipe_parent, generation), daemon=True
        ).start()

    def _listen_pipe(self, conn: Connection, generation: int):
        while generation == self._listener_generation:
            try:
                if conn.poll(0.1):
                    msg = conn.recv()
                    if self.app_ref:
                        self.app_ref.page.pubsub.send_all(msg)
            except (EOFError, OSError):
                if generation == self._listener_generation and self.app_ref:
                    self.app_ref.page.pubsub.send_all(
                        MotorMsgError("Pipe do motor encerrado inesperadamente.")
                    )
                return

    def main_flet(self, page: ft.Page):
        async def on_window_event(e: ft.WindowEvent):
            if e.type == ft.WindowEventType.CLOSE:
                if self.pipe_parent:
                    try:
                        self.pipe_parent.send(UIMsgShutdown())
                    except Exception:
                        pass
                await page.window.destroy()

        page.window.prevent_close = True
        page.window.on_event = on_window_event

        # main_flet só constrói uma VTEApp nova; nunca lança uma thread de
        # leitura (isso é responsabilidade exclusiva desta classe, acima).
        self.app_ref = VTEApp(
            page, restart_callback=self.start_motor, context_length=self.context_length
        )
        self.app_ref.reset_for_new_motor(self.pipe_parent, self.context_length)


class VTEApp:
    def __init__(self, page: ft.Page, restart_callback, context_length: int):
        self.page = page
        self.restart_callback = restart_callback
        self.current_context_length = context_length
        self.mode = "dark"
        self.palette = get_palette(self.mode)

        self.pipe_conn: Connection = None
        self.current_reply: ft.Text = None
        self.log_lines = []

        page.title = "VTE — RDNA3 Inference Engine"
        page.padding = 0
        page.theme_mode = ft.ThemeMode.DARK
        page.window.width = 1180
        page.window.height = 780
        page.window.min_width = 860
        page.window.min_height = 560
        page.window.icon = "earth_pixelated.svg"

        self._build_ui()
        page.add(self.progress_screen, self.main_layout)
        page.pubsub.subscribe(self._on_motor_message)
        self._apply_palette()

    # ------------------------------------------------------------------
    # Construção da UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        self._build_topbar()
        self._build_chat_panel()
        self._build_dashboard_panel()
        self._build_progress_screen()
        self._build_crash_banner()

        self.main_layout = ft.Column(
            [
                self.topbar,
                ft.Row(
                    [
                        ft.Container(self.chat_panel, expand=True, padding=16),
                        ft.Container(self.dashboard_panel, padding=16, width=340),
                    ],
                    expand=True,
                    spacing=0,
                ),
            ],
            expand=True,
            visible=False,
            spacing=0,
        )

    def _build_topbar(self):
        self.app_icon = ft.Image(src="earth_pixelated.svg", width=24, height=24)
        self.title_text = ft.Text(
            "VTE", size=18, weight=ft.FontWeight.BOLD
        )
        self.subtitle_text = ft.Text("RDNA3 Inference Engine", size=12)

        self.context_dropdown = ft.Dropdown(
            label="Context size",
            width=140,
            value=str(self.current_context_length),
            options=[ft.dropdown.Option(str(v)) for v in CONTEXT_LENGTH_OPTIONS],
            tooltip="Requer reiniciar o motor (KV cache é dimensionado no carregamento do modelo)",
            on_select=self._on_context_change,
            disabled=True,
        )

        self.theme_btn = ft.IconButton(
            icon=ft.Icons.LIGHT_MODE,
            tooltip="Alternar tema claro/escuro",
            on_click=self._toggle_theme,
        )
        self.restart_btn = ft.IconButton(
            icon=ft.Icons.RESTART_ALT,
            tooltip="Reiniciar motor",
            on_click=lambda e: self.restart_callback(),
        )

        self.status_pill = ft.Container(
            content=ft.Text("inicializando...", size=12, weight=ft.FontWeight.W_600),
            padding=ft.Padding(left=10, right=10, top=4, bottom=4),
            border_radius=12,
        )

        self.topbar = ft.Container(
            content=ft.Row(
                [
                    ft.Row(
                        [self.app_icon, self.title_text, self.subtitle_text, self.status_pill],
                        spacing=10,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Row(
                        [self.context_dropdown, self.theme_btn, self.restart_btn],
                        spacing=4,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            padding=ft.Padding(left=20, right=16, top=10, bottom=10),
        )

    def _build_chat_panel(self):
        self.chat_list = ft.ListView(expand=True, spacing=10, auto_scroll=True)
        self.input_field = ft.TextField(
            hint_text="Digite o prompt para o Qwen2.5...",
            expand=True,
            border_radius=8,
            on_submit=self._handle_submit,
        )
        self.send_btn = ft.IconButton(icon=ft.Icons.SEND, on_click=self._handle_submit)
        self.cancel_btn = ft.IconButton(
            icon=ft.Icons.STOP_CIRCLE, visible=False, on_click=self._handle_cancel
        )

        self.chat_panel = ft.Column(
            [
                self.chat_list,
                ft.Row([self.input_field, self.send_btn, self.cancel_btn]),
            ],
            expand=True,
        )

    def _build_dashboard_panel(self):
        def metric_label(text):
            return ft.Text(text, size=11)

        self.temp_value = ft.Text("0 °C", size=22, weight=ft.FontWeight.BOLD)
        self.tps_value = ft.Text("0.0 tok/s", size=22, weight=ft.FontWeight.BOLD)
        self.ms_value = ft.Text("0.0 ms/tok", size=13)
        self.vram_value = ft.Text("0 MB", size=20, weight=ft.FontWeight.BOLD)

        self.vram_weights_text = ft.Text("Weights: 0 MB", size=11)
        self.vram_kv_text = ft.Text("KV Cache: 0 MB", size=11)
        self.vram_arena_text = ft.Text("Arena: 0 MB", size=11)
        # "Sistema (dedicada)" é o número que bate com o Gerenciador de
        # Tarefas (WMI GPUAdapterMemory) -- inclui OUTROS processos (desktop,
        # navegador etc.), por isso fica separado do valor grande acima
        # (que é só o que o VTE mesmo alocou, determinístico). Misturar os
        # dois fazia o número principal saltar de forma confusa por
        # atividade alheia ao VTE.
        self.vram_system_text = ft.Text("Sistema (dedicada): — GB", size=11)
        self.vram_free_text = ft.Text("Livre p/ sistema: 0 GB", size=11, weight=ft.FontWeight.W_600)
        self.vram_details_col = ft.Column(
            [self.vram_weights_text, self.vram_kv_text, self.vram_arena_text,
             ft.Container(height=4), self.vram_system_text, self.vram_free_text],
            spacing=2, visible=False,
        )

        self.lifecycle_text = ft.Text("Inicializando...", size=13)

        self._metric_labels = []

        def metric_block(label_text, *value_controls):
            lbl = metric_label(label_text)
            self._metric_labels.append(lbl)
            return ft.Column([lbl, *value_controls], spacing=2)

        metrics_tab = ft.Container(
            content=ft.Column(
                [
                    metric_block("TEMPERATURA", self.temp_value),
                    metric_block("VELOCIDADE DE INFERÊNCIA", self.tps_value, self.ms_value),
                    metric_block("VRAM (ALOCADA PELO VTE)", self.vram_value, self.vram_details_col),
                    metric_block("CICLO DE VIDA DO MODELO", self.lifecycle_text),
                ],
                spacing=18,
            ),
            padding=16,
        )

        self.log_list = ft.ListView(expand=True, spacing=1, auto_scroll=True)
        logs_tab = ft.Container(content=self.log_list, padding=10, expand=True)

        metrics_tab.visible = True
        logs_tab.visible = False
        self.metrics_view = metrics_tab
        self.logs_view = logs_tab
        self._active_dash_tab = "metrics"

        # ft.Tabs nesta versão do Flet não tem um slot de conteúdo por aba
        # (Tab só carrega o label) -- em vez de brigar com essa API nova e
        # pouco documentada, um par de botões + troca manual de visível
        # resolve exatamente o que precisamos aqui (2 painéis, nunca mais).
        self.tab_metrics_btn = ft.TextButton("Métricas", on_click=lambda e: self._select_dash_tab("metrics"))
        self.tab_logs_btn = ft.TextButton("Logs", on_click=lambda e: self._select_dash_tab("logs"))

        self.dashboard_panel = ft.Container(
            content=ft.Column(
                [
                    ft.Row([self.tab_metrics_btn, self.tab_logs_btn], spacing=4),
                    ft.Divider(height=1),
                    ft.Column([self.metrics_view, self.logs_view], expand=True),
                ],
                expand=True,
                spacing=0,
            ),
            border_radius=10,
            expand=True,
        )

    def _build_progress_screen(self):
        self.progress_title = ft.Text("Inicializando Motor VTE", size=22, weight=ft.FontWeight.BOLD)
        self.progress_status = ft.Text("Preparando HIP Runtime...", size=13)
        self.progress_bar = ft.ProgressBar(width=420)

        self.progress_screen = ft.Container(
            content=ft.Column(
                [self.progress_title, self.progress_bar, self.progress_status],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=16,
            ),
            expand=True,
            alignment=ft.Alignment.CENTER,
        )

    def _build_crash_banner(self):
        self.tdr_banner_detail = ft.Text("", size=11, italic=True)
        self.tdr_banner = ft.Banner(
            leading=ft.Icon(ft.Icons.WARNING_AMBER_ROUNDED, size=40),
            content=ft.Column(
                [
                    ft.Text(
                        "Núcleo da GPU abortado (proteção WDDM/TDR ativada, ou o motor "
                        "encerrou de forma inesperada). O contexto do driver foi reiniciado."
                    ),
                    self.tdr_banner_detail,
                ],
                spacing=4, tight=True,
            ),
            actions=[ft.TextButton("Reiniciar Motor", on_click=self._restart_from_banner)],
        )

    # ------------------------------------------------------------------
    # Tema
    # ------------------------------------------------------------------
    def _toggle_theme(self, e):
        self.mode = "light" if self.mode == "dark" else "dark"
        self.palette = get_palette(self.mode)
        self.page.theme_mode = ft.ThemeMode.LIGHT if self.mode == "light" else ft.ThemeMode.DARK
        self._apply_palette()

    def _apply_palette(self):
        p = self.palette
        page = self.page

        page.bgcolor = p.bg
        self.title_text.color = p.accent_green
        self.subtitle_text.color = p.text_muted
        self.theme_btn.icon = ft.Icons.DARK_MODE if self.mode == "light" else ft.Icons.LIGHT_MODE
        self.theme_btn.icon_color = p.text_muted
        self.restart_btn.icon_color = p.text_muted
        self.topbar.bgcolor = p.panel
        self.topbar.border = ft.Border(bottom=ft.BorderSide(1, p.border))

        self.context_dropdown.border_color = p.border
        self.context_dropdown.color = p.text_primary
        self.context_dropdown.bgcolor = p.input_bg
        self.context_dropdown.label_style = ft.TextStyle(color=p.text_muted)

        # Chat
        self.input_field.border_color = p.border
        self.input_field.focused_border_color = p.accent_green
        self.input_field.cursor_color = p.accent_green
        self.input_field.bgcolor = p.input_bg
        self.input_field.color = p.text_primary
        self.input_field.hint_style = ft.TextStyle(color=p.text_muted)
        self.send_btn.icon_color = p.accent_green
        self.cancel_btn.icon_color = p.accent_red

        # Dashboard
        self.dashboard_panel.bgcolor = p.panel
        self.dashboard_panel.border = ft.Border.all(1, p.border)
        for lbl in self._metric_labels:
            lbl.color = p.text_muted
        self.tps_value.color = p.accent_green
        self.ms_value.color = p.text_muted
        self.vram_value.color = p.text_primary
        self.vram_weights_text.color = p.text_muted
        self.vram_kv_text.color = p.text_muted
        self.vram_arena_text.color = p.text_muted
        self.vram_system_text.color = p.text_muted
        self.vram_free_text.color = p.accent_green
        self.lifecycle_text.color = p.text_muted
        if self.temp_value.color not in (p.accent_red,):
            self.temp_value.color = p.text_primary
        self._style_dash_tab_buttons()

        # Progress screen
        self.progress_screen.bgcolor = p.bg
        self.progress_title.color = p.text_primary
        self.progress_status.color = p.text_muted
        self.progress_bar.color = p.accent_green
        self.progress_bar.bgcolor = p.border

        # Banner
        self.tdr_banner.bgcolor = p.accent_red
        self.tdr_banner.leading.color = p.text_strong
        for line in self.tdr_banner.content.controls:
            line.color = p.text_strong
        self.tdr_banner.actions[0].style = ft.ButtonStyle(color=p.text_strong)

        self._recolor_log_lines()
        self._recolor_status_pill()

        if page.controls:
            page.update()

    def _recolor_status_pill(self):
        p = self.palette
        text_ctrl: ft.Text = self.status_pill.content
        if "ativo" in text_ctrl.value.lower() or "carregado" in text_ctrl.value.lower():
            self.status_pill.bgcolor = ft.Colors.with_opacity(0.14, p.accent_green)
            text_ctrl.color = p.accent_green
        else:
            self.status_pill.bgcolor = p.border
            text_ctrl.color = p.text_muted

    def _select_dash_tab(self, name: str):
        self._active_dash_tab = name
        self.metrics_view.visible = (name == "metrics")
        self.logs_view.visible = (name == "logs")
        self._style_dash_tab_buttons()
        self.dashboard_panel.update()

    def _style_dash_tab_buttons(self):
        p = self.palette
        active = p.accent_green
        inactive = p.text_muted
        self.tab_metrics_btn.style = ft.ButtonStyle(
            color=active if self._active_dash_tab == "metrics" else inactive
        )
        self.tab_logs_btn.style = ft.ButtonStyle(
            color=active if self._active_dash_tab == "logs" else inactive
        )

    def _recolor_log_lines(self):
        p = self.palette
        for row, level in zip(self.log_list.controls, (l.level for l in self.log_lines)):
            row.color = self._log_level_color(level, p)

    def _log_level_color(self, level: str, p: Palette) -> str:
        if level in ("ERROR", "CRITICAL"):
            return p.accent_red
        if level == "WARNING":
            return p.accent_blue
        return p.text_muted

    # ------------------------------------------------------------------
    # Pipe / IPC (thread de fundo -> pubsub -> handler async no event loop)
    # ------------------------------------------------------------------
    def reset_for_new_motor(self, pipe_conn: Connection, context_length: int):
        """Chamado pelo Orchestrator sempre que um processo de motor novo
        nasce (boot inicial, crash recovery, ou troca de context_length).
        Só atualiza estado/UI -- a leitura do pipe é responsabilidade
        exclusiva do Orchestrator (ver seu docstring de classe)."""
        self.pipe_conn = pipe_conn
        self.current_context_length = context_length
        self.context_dropdown.value = str(context_length)
        self.context_dropdown.disabled = True

        self.main_layout.visible = False
        self.progress_screen.visible = True
        self.progress_bar.value = 0.0
        self.progress_status.value = "Preparando HIP Runtime..."
        self.chat_list.controls.clear()
        self.current_reply = None
        self.page.update()

    async def _on_motor_message(self, msg):
        """Roda no event loop da sessão (despachado via
        run_coroutine_threadsafe pelo PubSubHub) — seguro para mutar
        controles e chamar .update() aqui."""
        if isinstance(msg, MotorMsgProgress):
            self.progress_status.value = msg.status
            self.progress_bar.value = msg.percentage / 100.0
            self.progress_screen.update()

        elif isinstance(msg, MotorMsgReady):
            self.progress_screen.visible = False
            self.main_layout.visible = True
            self.context_dropdown.disabled = False
            self.page.update()

        elif isinstance(msg, MotorMsgToken):
            self._append_token(msg.text)

        elif isinstance(msg, MotorMsgDone):
            self._end_generation()

        elif isinstance(msg, MotorMsgMetrics):
            self._update_metrics(msg)

        elif isinstance(msg, MotorMsgStatusUpdate):
            self._update_lifecycle(msg)

        elif isinstance(msg, MotorMsgLog):
            self._append_log(msg)

        elif isinstance(msg, MotorMsgError):
            self._show_crash(msg.message)

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------
    def _handle_submit(self, e):
        if not self.input_field.value or not self.pipe_conn:
            return

        prompt = self.input_field.value
        self.input_field.value = ""
        self.input_field.disabled = True
        self.context_dropdown.disabled = True
        self.send_btn.visible = False
        self.cancel_btn.visible = True
        self.page.update()

        user_text = ft.Text(prompt, color=self.palette.text_primary, selectable=True)
        self._fit_bubble_text(user_text)
        self.chat_list.controls.append(self._make_bubble(user_text, is_user=True))

        self.current_reply = ft.Text("", color=self.palette.accent_green, selectable=True)
        self.chat_list.controls.append(self._make_bubble(self.current_reply, is_user=False))
        self.page.update()
        self.pipe_conn.send(UIMsgPrompt(text=prompt))

    # Acima deste tamanho o texto ganha uma largura máxima (e passa a
    # quebrar linha); abaixo, o Text fica sem `width` e a bolha simplesmente
    # abraça o conteúdo (uma linha só). Ajustado para caber ~1 linha de "oi"/
    # "Hola! Como posso ajudar?" sem cortar, mas capar respostas de parágrafo.
    _BUBBLE_WRAP_THRESHOLD_CHARS = 42
    _BUBBLE_MAX_WIDTH = 420

    def _fit_bubble_text(self, text_ctrl: ft.Text):
        """Decide se o Text da bolha deve abraçar o conteúdo (width=None,
        uma linha) ou ganhar uma largura máxima e quebrar (mensagens longas).
        Container não tem max_width nesta versão do Flet -- por isso o teto
        é aplicado no próprio Text, que é quem efetivamente controla onde a
        linha quebra; a bolha (Container) sem `expand` só abraça o que o Text
        ocupar."""
        text_ctrl.width = self._BUBBLE_MAX_WIDTH if len(text_ctrl.value or "") > self._BUBBLE_WRAP_THRESHOLD_CHARS else None

    def _make_bubble(self, content: ft.Control, is_user: bool) -> ft.Row:
        """Bolha de chat que abraça o tamanho do conteúdo (curto = uma linha
        compacta) e só cresce até `_BUBBLE_MAX_WIDTH` para mensagens longas,
        quebrando o texto -- nunca estica para 75% do painel só porque a
        mensagem é um "oi"."""
        p = self.palette
        bubble = ft.Container(
            content=content,
            padding=10,
            border_radius=10,
            bgcolor=p.input_bg if is_user else p.panel,
            border=None if is_user else ft.Border.all(1, p.border),
        )
        return ft.Row(
            [bubble],
            alignment=ft.MainAxisAlignment.END if is_user else ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )

    def _handle_cancel(self, e):
        if self.pipe_conn:
            self.pipe_conn.send(UIMsgCancel())

    def _append_token(self, token: str):
        if self.current_reply:
            self.current_reply.value += token
            self._fit_bubble_text(self.current_reply)
            self.current_reply.update()

    def _end_generation(self):
        self.input_field.disabled = False
        self.context_dropdown.disabled = False
        self.send_btn.visible = True
        self.cancel_btn.visible = False
        self.current_reply = None
        self.page.update()

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------
    def _update_metrics(self, msg: MotorMsgMetrics):
        p = self.palette
        if msg.temp_c is None:
            # Honesto em vez de inventado: não há fonte de temperatura real
            # da GPU disponível via WMI padrão sem o SDK da AMD (ADL/ADLX) --
            # ver gpu_monitor.py. Mostrar um número aqui pareceria real sem
            # ser.
            self.temp_value.value = "N/A"
            self.temp_value.color = p.text_muted
        else:
            self.temp_value.value = f"{msg.temp_c:.1f} °C"
            self.temp_value.color = p.accent_red if msg.temp_c > 85.0 else p.text_primary

        self.tps_value.value = f"{msg.tokens_sec:.1f} tok/s"
        self.ms_value.value = f"{msg.ms_per_token:.1f} ms/tok" if msg.ms_per_token > 0 else "—"

        self.vram_value.value = f"{msg.vram_mb:.0f} MB"

        if msg.vram_details and msg.vram_details.get('total_mb', 0) > 0:
            d = msg.vram_details
            self.vram_weights_text.value = f"Weights: {d.get('weights_mb', 0):.0f} MB"
            self.vram_kv_text.value = f"KV Cache: {d.get('kv_cache_mb', 0):.0f} MB"
            self.vram_arena_text.value = f"Arena: {d.get('activations_mb', 0):.0f} MB"
            if msg.system_dedicated_vram_mb > 0:
                self.vram_system_text.value = f"Sistema (dedicada): {msg.system_dedicated_vram_mb / 1024:.1f} GB"
            else:
                self.vram_system_text.value = "Sistema (dedicada): —"
            self.vram_free_text.value = f"Livre p/ sistema: {msg.vram_free_system_mb / 1024:.1f} GB"
            self.vram_details_col.visible = True
        else:
            self.vram_details_col.visible = False

        self.dashboard_panel.update()

    def _update_lifecycle(self, msg: MotorMsgStatusUpdate):
        p = self.palette
        if not msg.is_loaded:
            self.lifecycle_text.value = "Modelo descarregado (ocioso)"
            self.lifecycle_text.color = p.text_muted
            self.status_pill.content.value = "ocioso"
        else:
            if msg.time_until_unload is not None:
                self.lifecycle_text.value = f"Ativo (unload em {msg.time_until_unload:.0f}s)"
            else:
                self.lifecycle_text.value = "Ativo"
            self.lifecycle_text.color = p.accent_green
            self.status_pill.content.value = "ativo"
        self._recolor_status_pill()
        self.status_pill.update()
        self.lifecycle_text.update()

    def _append_log(self, msg: MotorMsgLog):
        self.log_lines.append(msg)
        if len(self.log_lines) > MAX_LOG_LINES:
            self.log_lines.pop(0)
            if self.log_list.controls:
                self.log_list.controls.pop(0)

        row = ft.Text(
            msg.text, size=11, font_family="Consolas",
            color=self._log_level_color(msg.level, self.palette),
            selectable=True,
        )
        self.log_list.controls.append(row)
        self.log_list.update()

    # ------------------------------------------------------------------
    # Context length / restart
    # ------------------------------------------------------------------
    def _on_context_change(self, e):
        try:
            new_length = int(self.context_dropdown.value)
        except (TypeError, ValueError):
            return
        if new_length == self.current_context_length:
            return
        self.restart_callback(context_length=new_length)

    def _restart_from_banner(self, e):
        self.page.pop_dialog()
        self.restart_callback()

    def _show_crash(self, message: str):
        self._end_generation()
        self.tdr_banner_detail.value = message or ""
        try:
            self.page.show_dialog(self.tdr_banner)
        except RuntimeError:
            # Já aberto -- pode acontecer se dois MotorMsgError chegarem em
            # sequência (ex.: falha no boot() seguida do listener detectando
            # o pipe morto). Só atualiza o detalhe, não reabre.
            self.tdr_banner.update()


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

    ft.run(main=orchestrator.main_flet, view=ft.AppView.FLET_APP, assets_dir=ASSETS_DIR)


if __name__ == '__main__':
    cli_main()
