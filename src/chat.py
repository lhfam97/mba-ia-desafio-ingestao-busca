from search import search_prompt


def print_welcome_message():    
    print("=" * 60)
    print("💬 Chat iniciado!")
    print("Powered by OpenApi + LangChain + pgVector")
    print("=" * 60)
    print("Digite 'fechar', 'exit' ou 'close' para encerrar.")

def main():
    print_welcome_message();

    while True:
            mensagem = input("Você: ").strip().lower()  # remove espaços e transforma em minúsculas
            if mensagem in ("fechar", "exit", "close"):
                print("👋 Chat encerrado. Até mais!")
                break
            else:
                message = search_prompt(mensagem)
                print(f"Bot: '{message}'")
        # chain = search_prompt()

        # if not chain:
        #     print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        #     return
        
        # pass

if __name__ == "__main__":
    main()