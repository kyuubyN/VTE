import logging
logging.basicConfig(level=logging.INFO)

from vte.core.model import VTEModel

def main():
    print("Carregando modelo...")
    model = VTEModel.from_pretrained("qwen2.5:1.5b-q4_k_m", use_hip_graph=False, enable_fusion=True)
    
    print("\nIniciando geração:")
    prompt = "Explique inteligência artificial em uma frase"
    print(f"Prompt: {prompt}\nResponse: ", end="")
    
    try:
        for token in model.generate(prompt, max_tokens=15):
            print(token, end="", flush=True)
    except Exception as e:
        print(f"\n[ERRO DURANTE A GERAÇÃO] {e}")
        import traceback
        traceback.print_exc()
        
    print("\n\nTeste concluído. Fechando.")

if __name__ == "__main__":
    main()
