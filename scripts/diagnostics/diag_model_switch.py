"""
Reproduz o caminho que o botao de troca de modelo do Flet dispara
(Orchestrator.start_motor(model_name=...)) sem precisar de UI real -- teste
de regressao para a corrida de geracao (mensagens do motor antigo vazando
pro pubsub) encontrada e corrigida nesta sessao.
"""
import multiprocessing
import time
from vte.ui.app import Orchestrator
from vte.core.ipc import MotorMsgReady, MotorMsgError, MotorMsgLog


class FakeAppRef:
    class FakePage:
        class FakePubSub:
            def send_all(self, msg):
                if isinstance(msg, MotorMsgLog):
                    return
                print(f"[PUBSUB] {type(msg).__name__}: {msg}")
        pubsub = FakePubSub()
    page = FakePage()

    def reset_for_new_motor(self, pipe_conn, context_length, model_name):
        print(f"[APP] reset_for_new_motor(context_length={context_length}, model_name={model_name})")


def main():
    o = Orchestrator()
    o.app_ref = FakeAppRef()

    print("=== 1) start_motor() -- Qwen default ===")
    o.start_motor()

    print("Aguardando MotorMsgReady...")
    ready = False
    t0 = time.time()
    while time.time() - t0 < 60:
        if o._motor_ready:
            ready = True
            break
        time.sleep(0.2)
    print(f"Ready={ready} apos {time.time()-t0:.1f}s")

    if not ready:
        print("NUNCA FICOU PRONTO -- abortando reproducao")
        raise SystemExit(1)

    print("\n=== 2) start_motor(model_name='granite-4.1:3b-q8_0') -- TROCA DE MODELO ===")
    time.sleep(2)
    o.start_motor(model_name="granite-4.1:3b-q8_0")

    print("Aguardando MotorMsgReady do Granite...")
    ready2 = False
    saw_error = False
    t0 = time.time()
    while time.time() - t0 < 90:
        if o._motor_ready:
            ready2 = True
            break
        if o.motor_process and not o.motor_process.is_alive():
            print(f"!!! Processo do motor MORREU (exitcode={o.motor_process.exitcode}) antes de ficar pronto !!!")
            break
        time.sleep(0.2)
    print(f"Ready (Granite)={ready2} apos {time.time()-t0:.1f}s")

    print("\nResultado final:", "SUCESSO" if ready2 else "FALHOU")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
