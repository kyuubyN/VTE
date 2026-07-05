"""
diag_thinking_mode.py -- valida o mecanismo de "Thinking mode"
(vte/core/thinking_scanner.py) em duas partes:

1) Unitária (sem GPU): casos sintéticos, incluindo tag <think>/</think>
   cortada no meio por um flush -- roda em milissegundos.
2) Ponta a ponta com geração REAL (Qwen2.5 ou Granite, o que já estiver
   disponível em Model/): nenhum dos dois tem <think> como token especial
   treinado, então o teste instrui o modelo via prompt a usar essas tags
   como texto comum -- valida que o SCANNER (que opera sobre o texto
   decodificado, não sobre IDs de token especiais) separa as seções
   corretamente mesmo assim, com uma geração real na GPU.

Mede tok/s antes/depois de passar pelo scanner para confirmar que ele não
introduz overhead perceptível (disciplina "medir, não supor" do projeto --
ver docs/BUGS.md).
"""
import time

from vte.core.thinking_scanner import ThinkingSectionScanner


def test_unit_cases():
    print("=== 1) Casos sintéticos (unitário, sem GPU) ===")

    cases = [
        ("sem tags nenhuma", ["Isso é uma resposta comum, sem pensamento algum."]),
        ("tag completa num chunk só", ["<think>raciocinando</think>Resposta final."]),
        ("tag cortada no meio por um flush", ["texto antes <thi", "nk>pensando", "</thi", "nk>depois"]),
        ("múltiplas transições", ["a<think>p1</think>b<think>p2</think>c"]),
    ]

    all_ok = True
    for name, chunks in cases:
        scanner = ThinkingSectionScanner()
        out = []
        for c in chunks:
            out.extend(scanner.feed(c))
        out.extend(scanner.flush())

        answer = "".join(s.text for s in out if s.section == "answer")
        thinking = "".join(s.text for s in out if s.section == "thinking")
        # nenhuma tag literal deve sobrar no texto final (foram engolidas)
        leaked_tag = "<think>" in answer + thinking or "</think>" in answer + thinking
        status = "FALHOU (tag vazou)" if leaked_tag else "ok"
        if leaked_tag:
            all_ok = False
        print(f"  [{status}] {name}: answer={answer!r} thinking={thinking!r}")

    print(f"\nResultado unitário: {'TODOS OK' if all_ok else 'HÁ FALHAS'}")
    return all_ok


def _timed_generate(model, chat_prompt, max_tokens, use_scanner):
    """Gera e mede tok/s EXCLUINDO o primeiro token (prefill do prompt
    inteiro, não custo de decode por token -- mesma convenção já usada em
    motor.py::generate(), que pula o delta do 1o token pelo mesmo motivo:
    misturar isso infla falsamente o ms/token médio, principalmente em
    gerações curtas onde o prefill domina a amostra."""
    scanner = ThinkingSectionScanner() if use_scanner else None
    thinking_parts, answer_parts = [], []
    n_tokens = 0
    last_t = None
    decode_elapsed = 0.0

    for word in model.generate(chat_prompt, max_tokens=max_tokens):
        now = time.perf_counter()
        if last_t is not None:
            decode_elapsed += now - last_t
        last_t = now
        n_tokens += 1

        if scanner is not None:
            for chunk in scanner.feed(word):
                (thinking_parts if chunk.section == "thinking" else answer_parts).append(chunk.text)
        else:
            answer_parts.append(word)

    if scanner is not None:
        for chunk in scanner.flush():
            (thinking_parts if chunk.section == "thinking" else answer_parts).append(chunk.text)

    tok_s = (n_tokens - 1) / decode_elapsed if decode_elapsed > 0 else 0.0
    return {
        "n_tokens": n_tokens,
        "tok_s": tok_s,
        "thinking": "".join(thinking_parts),
        "answer": "".join(answer_parts),
    }


def test_e2e_real_model():
    print("\n=== 2) Ponta a ponta com modelo real ===")
    from vte.core.model import VTEModel

    model_name = "qwen2.5:1.5b-q4_k_m"
    print(f"Carregando {model_name}...")
    model = VTEModel.from_pretrained(model_name, context_length=1024)

    prompt = (
        "Antes de responder, pense em voz alta dentro de tags <think> e "
        "</think>. Depois dessas tags, escreva sua resposta final normalmente. "
        "Pergunta: quanto é 12 vezes 7?"
    )
    chat_prompt = model.tokenizer.apply_chat_template(prompt, enable_thinking=True)

    print("\n--- Rodada A: COM o ThinkingSectionScanner ---")
    r_with = _timed_generate(model, chat_prompt, max_tokens=200, use_scanner=True)
    print(f"Tokens: {r_with['n_tokens']} | tok/s (sem prefill): {r_with['tok_s']:.1f}")
    print(f"Thinking capturado: {r_with['thinking'].strip()[:300]!r}")
    print(f"Answer capturado: {r_with['answer'].strip()[:300]!r}")

    print("\n--- Rodada B: SEM o scanner (baseline, texto cru) ---")
    r_without = _timed_generate(model, chat_prompt, max_tokens=200, use_scanner=False)
    print(f"Tokens: {r_without['n_tokens']} | tok/s (sem prefill): {r_without['tok_s']:.1f}")

    got_thinking = bool(r_with["thinking"].strip())
    no_leaked_tags = "<think>" not in r_with["answer"] and "</think>" not in r_with["answer"]
    overhead_pct = (
        100.0 * (r_without["tok_s"] - r_with["tok_s"]) / r_without["tok_s"]
        if r_without["tok_s"] > 0 else float("nan")
    )

    print(f"\nSeparou pensamento de resposta: {got_thinking}")
    print(f"Nenhuma tag vazou pro texto final: {no_leaked_tags}")
    print(
        f"Overhead do scanner: {overhead_pct:+.1f}% "
        f"(com scanner: {r_with['tok_s']:.1f} tok/s, sem: {r_without['tok_s']:.1f} tok/s)"
    )
    print(
        "Baseline documentado do Qwen2.5 (decode HIP Graph, contexto de produção): ~100 tok/s -- "
        "esta medição usa um prompt/contexto diferente (curto, com instrução de thinking), "
        "então compare a DIFERENÇA relativa entre as rodadas A/B acima, não o valor absoluto "
        "contra os ~100 tok/s do benchmark oficial."
    )


if __name__ == "__main__":
    unit_ok = test_unit_cases()
    if not unit_ok:
        print("\nCasos unitários falharam -- pulando teste com modelo real.")
        raise SystemExit(1)
    test_e2e_real_model()
