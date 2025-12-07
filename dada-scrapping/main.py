import os
import urllib3
import time
from controller import CebraspeCrawler, PDFProcessor

# Suprime avisos de certificado SSL (limpa o terminal)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def resetar_banco_dados(nome_banco="concursos_data.db"):
    """Apaga o arquivo do banco se existir para começar do zero"""
    if os.path.exists(nome_banco):
        try:
            os.remove(nome_banco)
            print(f"🧹 Banco de dados antigo '{nome_banco}' apagado com sucesso.")
        except PermissionError:
            print(f"❌ Erro: O arquivo '{nome_banco}' está aberto. Feche o DB Browser e tente de novo.")
            exit()
    else:
        print(f"🆕 Criando novo banco de dados '{nome_banco}'...")

def extrair_nome_concurso(url):
    partes = url.strip('/').split('/')
    return partes[-1]

if __name__ == "__main__":
    # --- CONFIGURAÇÕES ---
    URL_ENCERRADOS = "https://www.cebraspe.org.br/concursos/encerrado"
    ARQUIVO_BANCO = "concursos_data.db"
    
    # --- MODO PRODUÇÃO TOTAL ---
    # None = Sem limite (faz tudo). 
    # Coloque um número (ex: 5) apenas se quiser testar rápido.
    LIMITE_TESTE = None 

    print("==================================================")
    print("   ROBÔ DE GABARITOS CEBRASPE - VERSÃO FINAL      ")
    print("==================================================\n")

    # 1. Limpeza Inicial
    resetar_banco_dados(ARQUIVO_BANCO)
    
    crawler = CebraspeCrawler()
    processor = PDFProcessor()

    # 2. Obter a Lista Mestra
    print(f"📡 Acessando a lista de concursos encerrados...")
    lista_concursos = crawler.listar_todos_concursos(URL_ENCERRADOS)

    if not lista_concursos:
        print("❌ Erro fatal: Nenhum concurso encontrado. Verifique sua conexão.")
        exit()

    # Aplica o limite se houver (para testes)
    if LIMITE_TESTE:
        print(f"⚠️  MODO TESTE ATIVADO: Processando apenas {LIMITE_TESTE} concursos.")
        lista_concursos = lista_concursos[:LIMITE_TESTE]
    else:
        print(f"🚀 MODO PRODUÇÃO: Processando TODOS os {len(lista_concursos)} concursos.")

    # 3. Loop Principal
    total = len(lista_concursos)
    start_time = time.time()

    print(f"\nIniciando a maratona em 3, 2, 1...\n")

    for i, url_concurso in enumerate(lista_concursos, 1):
        nome_concurso = extrair_nome_concurso(url_concurso)
        
        # Cabeçalho visual para acompanhar o progresso
        print(f"--------------------------------------------------")
        print(f"PROJETO [{i}/{total}]: {nome_concurso}")
        print(f"URL: {url_concurso}")
        
        try:
            # A. Mapear Cargos
            mapa_cargos = crawler.mapear_cargos(url_concurso)

            if not mapa_cargos:
                print(f"   ⚠️  Nenhum gabarito definitivo encontrado. Pulando.")
                continue

            # B. Processar Cargos
            print(f"   🔎 Encontrados {len(mapa_cargos)} grupos de cargos.")
            
            for id_cargo, links in mapa_cargos.items():
                processor.limpar_memoria()
                
                # Download e Leitura
                if 'basico' in links:
                    processor.processar_pdf(links['basico'], nome_concurso, "Conhec. Básicos")
                
                if 'especifico' in links:
                    processor.processar_pdf(links['especifico'], nome_concurso, "Conhec. Específicos")

                # Salvar no SQLite
                processor.salvar_final(nome_concurso, id_cargo)

        except KeyboardInterrupt:
            print("\n🛑 Processo interrompido pelo usuário.")
            break
        except Exception as e:
            print(f"❌ Erro crítico no concurso {nome_concurso}: {e}")
            continue

        # Pequena pausa para o servidor respirar
        time.sleep(1)

    # 4. Relatório Final
    tempo_total = (time.time() - start_time) / 60
    print("\n==================================================")
    print("✅✅✅  COLETA FINALIZADA COM SUCESSO!  ✅✅✅")
    print(f"Tempo total: {tempo_total:.2f} minutos")
    print(f"Banco de dados gerado: {ARQUIVO_BANCO}")
    print("==================================================")