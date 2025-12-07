class TerminalView:
    def mostrar_inicio(self):
        print("--- Iniciando Coleta de Gabaritos ---")

    def mostrar_status(self, mensagem):
        print(f"[STATUS] {mensagem}")

    def mostrar_sucesso(self, questao):
        print(f"✅ Capturado: Questão {questao.numero_questao} - Letra {questao.alternativa_correta}")

    def mostrar_erro(self, erro):
        print(f"❌ Erro: {erro}")

    def mostrar_fim(self, total):
        print("-------------------------------------")
        print(f"Processo finalizado. Total de questões salvas: {total}")