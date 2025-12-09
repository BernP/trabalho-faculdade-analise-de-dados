import sqlite3
import pandas as pd
import numpy as np
import random
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# Suprime avisos
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

class EstrategiaChute:
    @staticmethod
    def menos_marcada(gabarito_parcial, opcoes_possiveis):
        """Preenche lacunas com a letra menos frequente no parcial."""
        marcados = [x for x in gabarito_parcial if x is not None]
        if not marcados:
            # Se não marcou nada, chuta aleatório (ou 'A', mas aleatório evita viés artificial aqui)
            return [random.choice(opcoes_possiveis) for _ in gabarito_parcial]
            
        contagem = {op: marcados.count(op) for op in opcoes_possiveis}
        min_valor = min(contagem.values())
        candidatas = [k for k, v in contagem.items() if v == min_valor]
        letra_escolhida = random.choice(candidatas)
        gabarito_final = [x if x is not None else letra_escolhida for x in gabarito_parcial]
        return gabarito_final

class SimuladorCurvas:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        # Carrega provas uma vez só para memória
        self.df_provas = self._obter_todas_provas()

    def _obter_todas_provas(self):
        query = """
        SELECT DISTINCT c.nome, cg.nome_cargo, cg.id, cg.tipo_prova
        FROM cargos cg
        JOIN concursos c ON cg.concurso_id = c.id
        """
        return pd.read_sql_query(query, self.conn)

    def _carregar_gabarito(self, cargo_id):
        query = "SELECT resposta FROM gabaritos WHERE cargo_id = ? AND resposta != 'X' ORDER BY numero_questao"
        cursor = self.conn.cursor()
        cursor.execute(query, (cargo_id,))
        return [row[0] for row in cursor.fetchall()]

    def _determinar_opcoes(self, tipo_prova, gabarito):
        if tipo_prova == 'CERTO_ERRADO':
            return ['C', 'E'], "CERTO_ERRADO"
        else:
            if 'E' in gabarito: return ['A', 'B', 'C', 'D', 'E'], "MULTIPLA_5"
            else: return ['A', 'B', 'C', 'D'], "MULTIPLA_4"

    def _gerar_prova_simulada(self, gabarito_real, opcoes, conhecimento, erro):
        total = len(gabarito_real)
        n_tentativas = int(total * conhecimento)
        indices_tentativa = np.random.choice(total, n_tentativas, replace=False)
        
        n_erros = int(n_tentativas * erro)
        indices_erros = np.random.choice(indices_tentativa, n_erros, replace=False)
        set_erros = set(indices_erros)
        set_tentativas = set(indices_tentativa)
        
        folha = [None] * total
        acertos_ini = 0
        erros_ini = 0
        
        for i in range(total):
            if i in set_tentativas:
                if i in set_erros:
                    erradas = [op for op in opcoes if op != gabarito_real[i]]
                    folha[i] = random.choice(erradas)
                    erros_ini += 1
                else:
                    folha[i] = gabarito_real[i]
                    acertos_ini += 1
        
        return folha, acertos_ini, erros_ini

    def _calcular_nota(self, gabarito_cand, gabarito_real, tipo_grupo):
        acertos = 0
        erros = 0
        for c, r in zip(gabarito_cand, gabarito_real):
            if c == r: acertos += 1
            else: erros += 1
            
        if tipo_grupo == "CERTO_ERRADO":
            return acertos - erros
        else:
            return acertos

    def executar_cenario(self, conhecimento, erro, n_simulacoes=1000):
        """Roda a simulação para UM par específico de Conhecimento/Erro"""
        resultados = []
        
        # Filtra provas vazias ou com problemas antes do loop
        # (Otimização básica)
        
        for idx, row in self.df_provas.iterrows():
            gabarito_real = self._carregar_gabarito(row['id'])
            if not gabarito_real: continue

            opcoes, grupo = self._determinar_opcoes(row['tipo_prova'], gabarito_real)
            
            # Acumuladores para média rápida (para não estourar memória com listas gigantes)
            soma_ganho = 0
            soma_nota_final = 0
            
            for _ in range(n_simulacoes):
                parcial, acertos_ini, erros_ini = self._gerar_prova_simulada(gabarito_real, opcoes, conhecimento, erro)
                
                # Nota Base (Se deixasse em branco)
                nota_base = (acertos_ini - erros_ini) if grupo == "CERTO_ERRADO" else acertos_ini
                
                # Chute
                final = EstrategiaChute.menos_marcada(parcial, opcoes)
                nota_final = self._calcular_nota(final, gabarito_real, grupo)
                
                soma_nota_final += nota_final
                soma_ganho += (nota_final - nota_base)

            # Salva média deste cenário para esta prova
            resultados.append({
                'Grupo': grupo,
                'Conhecimento_Pct': conhecimento,
                'Erro_Pct': erro,
                'Nota_Media_Final': soma_nota_final / n_simulacoes,
                'Ganho_Medio': soma_ganho / n_simulacoes
            })
            
        return pd.DataFrame(resultados)

    def gerar_estudo_curvas(self, lista_conhecimento, lista_erro, n_simulacoes=1000):
        """Loop Mestre que gera as curvas"""
        todos_resultados = []
        total_cenarios = len(lista_conhecimento) * len(lista_erro)
        contador = 1
        
        start_time = time.time()
        print(f"=== INICIANDO ESTUDO DE CURVAS ({total_cenarios} Cenários) ===")
        print(f"Simulações por Prova: {n_simulacoes}")
        
        for k in lista_conhecimento:
            for e in lista_erro:
                print(f"[{contador}/{total_cenarios}] Processando: Conhecimento {k*100:.0f}% | Erro {e*100:.0f}%...")
                
                df_cenario = self.executar_cenario(k, e, n_simulacoes)
                todos_resultados.append(df_cenario)
                contador += 1
                
        print(f"\nEstudo concluído em {(time.time() - start_time)/60:.2f} minutos.")
        
        # Consolida tudo num dataframe único
        return pd.concat(todos_resultados, ignore_index=True)

    def plotar_curvas(self, df_completo):
        """Gera os gráficos de linha"""
        grupos = df_completo['Grupo'].unique()
        
        print("\nGerando gráficos...")
        
        for grupo in grupos:
            df_g = df_completo[df_completo['Grupo'] == grupo]
            if df_g.empty: continue
            
            plt.figure(figsize=(10, 6))
            
            # O gráfico mostra: No Eixo X o conhecimento, No Y o ganho médio
            # As linhas coloridas (hue) são as taxas de erro
            sns.lineplot(
                data=df_g, 
                x='Conhecimento_Pct', 
                y='Ganho_Medio', 
                hue='Erro_Pct', 
                palette='viridis', 
                marker='o',
                linewidth=2.5
            )
            
            # Formatação
            plt.title(f'Curva de Eficiência do Chute: {grupo}', fontsize=14)
            plt.xlabel('Nível de Conhecimento do Candidato')
            plt.ylabel('Ganho Médio de Pontos (Pontos Extras)')
            
            # Formata Eixo X como %
            plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            
            # Legenda melhorada
            plt.legend(title='Taxa de Erro', title_fontsize='12')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            
            plt.tight_layout()
            plt.show()

class SimuladorHipotese:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.df_provas = self._obter_todas_provas()

    def _obter_todas_provas(self):
        query = """
        SELECT DISTINCT c.nome, cg.nome_cargo, cg.id, cg.tipo_prova
        FROM cargos cg
        JOIN concursos c ON cg.concurso_id = c.id
        """
        return pd.read_sql_query(query, self.conn)

    def _carregar_gabarito(self, cargo_id):
        query = "SELECT resposta FROM gabaritos WHERE cargo_id = ? AND resposta != 'X' ORDER BY numero_questao"
        cursor = self.conn.cursor()
        cursor.execute(query, (cargo_id,))
        return [row[0] for row in cursor.fetchall()]

    def _determinar_opcoes(self, tipo_prova, gabarito):
        if tipo_prova == 'CERTO_ERRADO':
            return ['C', 'E'], "CERTO_ERRADO"
        else:
            if 'E' in gabarito: return ['A', 'B', 'C', 'D', 'E'], "MULTIPLA_5"
            else: return ['A', 'B', 'C', 'D'], "MULTIPLA_4"

    def _gerar_cenario_inicial(self, gabarito_real, opcoes, conhecimento, erro):
        total = len(gabarito_real)
        n_tentativas = int(total * conhecimento)
        indices_tentativa = np.random.choice(total, n_tentativas, replace=False)
        
        n_erros = int(n_tentativas * erro)
        indices_erros = np.random.choice(indices_tentativa, n_erros, replace=False)
        set_erros = set(indices_erros)
        set_tentativas = set(indices_tentativa)
        
        folha = [None] * total
        acertos = 0
        erros = 0
        
        for i in range(total):
            if i in set_tentativas:
                if i in set_erros:
                    erradas = [op for op in opcoes if op != gabarito_real[i]]
                    folha[i] = random.choice(erradas)
                    erros += 1
                else:
                    folha[i] = gabarito_real[i]
                    acertos += 1
        return folha, acertos, erros

    def _calcular_nota_real(self, gabarito_cand, gabarito_real, tipo_grupo):
        acertos = 0
        erros = 0
        for c, r in zip(gabarito_cand, gabarito_real):
            if c == r: acertos += 1
            else: erros += 1
            
        if tipo_grupo == "CERTO_ERRADO":
            return acertos - erros
        else:
            return acertos

    def _calcular_nota_esperada_aleatoria(self, total, acertos_ini, erros_ini, grupo, n_opcoes):
        """Calcula a nota matemática esperada se chutasse aleatoriamente"""
        n_chutes = total - (acertos_ini + erros_ini)
        nota_base = acertos_ini - erros_ini if grupo == "CERTO_ERRADO" else acertos_ini
        
        if grupo == "CERTO_ERRADO":
            # Esperança matemática de chute em C/E é 0 (50% +1, 50% -1)
            return nota_base
        else:
            # Esperança matemática é (1/N_opcoes) por chute
            ganho_esperado = n_chutes * (1 / n_opcoes)
            return nota_base + ganho_esperado

    def executar_bateria_testes(self, niveis_conhecimento, erro_fixo, n_simulacoes=1000):
        print(f"=== BATERIA DE TESTES DE HIPÓTESE (Erro fixo: {erro_fixo*100}%) ===")
        print(f"Metodologia: Teste T Pareado (Comparando 'Técnica' vs 'Sorte' prova a prova)")
        print(f"Simulações por prova: {n_simulacoes}\n")
        
        # Estrutura para guardar resultados: dados[nivel][grupo] = {simulada: [], aleatoria: []}
        dados_coletados = {}

        # 1. COLETA DE DADOS (SIMULAÇÃO)
        total_provas = len(self.df_provas)
        print(f"Processando {total_provas} provas... (Aguarde)")
        
        for idx, row in self.df_provas.iterrows():
            gabarito_real = self._carregar_gabarito(row['id'])
            if not gabarito_real: continue

            opcoes, grupo = self._determinar_opcoes(row['tipo_prova'], gabarito_real)
            
            # Para cada nível de conhecimento, roda as simulações
            for nivel in niveis_conhecimento:
                if nivel not in dados_coletados: dados_coletados[nivel] = {}
                if grupo not in dados_coletados[nivel]: dados_coletados[nivel][grupo] = {'tec': [], 'rnd': []}
                
                soma_tec = 0
                soma_rnd = 0
                
                for _ in range(n_simulacoes):
                    parcial, acertos, erros = self._gerar_cenario_inicial(gabarito_real, opcoes, nivel, erro_fixo)
                    
                    # A. Nota com Técnica
                    final = EstrategiaChute.menos_marcada(parcial, opcoes)
                    nota_tec = self._calcular_nota_real(final, gabarito_real, grupo)
                    
                    # B. Nota com Sorte (Baseline)
                    nota_rnd = self._calcular_nota_esperada_aleatoria(len(gabarito_real), acertos, erros, grupo, len(opcoes))
                    
                    soma_tec += nota_tec
                    soma_rnd += nota_rnd
                
                # Guarda a MÉDIA desta prova para este nível
                dados_coletados[nivel][grupo]['tec'].append(soma_tec / n_simulacoes)
                dados_coletados[nivel][grupo]['rnd'].append(soma_rnd / n_simulacoes)

        # 2. EXECUÇÃO DOS TESTES ESTATÍSTICOS
        for nivel in niveis_conhecimento:
            print(f"\n" + "="*60)
            print(f"   CENÁRIO: CONHECIMENTO {nivel*100:.0f}% | ERRO {erro_fixo*100:.0f}%")
            print("="*60)
            
            for grupo in sorted(dados_coletados[nivel].keys()):
                vals_tec = dados_coletados[nivel][grupo]['tec']
                vals_rnd = dados_coletados[nivel][grupo]['rnd']
                
                if len(vals_tec) < 2:
                    print(f"Grupo {grupo}: Dados insuficientes.")
                    continue
                
                # Teste T Pareado
                stat, p_valor = stats.ttest_rel(vals_tec, vals_rnd, alternative='greater')
                
                media_tec = np.mean(vals_tec)
                media_rnd = np.mean(vals_rnd)
                diff = media_tec - media_rnd
                
                print(f"\n>> GRUPO: {grupo} ({len(vals_tec)} provas)")
                print(f"   Média Técnica: {media_tec:.2f} | Média Sorte: {media_rnd:.2f}")
                print(f"   Diferença (Ganho): {diff:+.2f} pontos")
                print(f"   Valor-P: {p_valor:.4e}")
                
                if p_valor < 0.05:
                    print("   [VÁLIDO] A técnica supera a sorte com significância estatística.")
                else:
                    print("   [INVÁLIDO] Não há evidência de que a técnica funcione melhor que o acaso.")

if __name__ == "__main__":
    DB_PATH = "../dada-scrapping/concursos_data.db"
    
    # --- CONFIGURAÇÃO DO ESTUDO ---
    LISTA_CONHECIMENTO = [0.50, 0.60, 0.70, 0.80, 0.90]
    LISTA_ERRO = [0.05, 0.10, 0.20]
    N_SIMULACOES = 1000 
    
    sim = SimuladorCurvas(DB_PATH)
    
    # 1. Gera os dados
    df_curvas = sim.gerar_estudo_curvas(LISTA_CONHECIMENTO, LISTA_ERRO, N_SIMULACOES)
    
    if not df_curvas.empty:
        # 2. Salva CSV
        df_curvas.to_csv("resultado_curvas_chute.csv", index=False)
        print("Dados salvos em 'resultado_curvas_chute.csv'")
        
        # 3. Plota
        sim.plotar_curvas(df_curvas)

        sim = SimuladorHipotese(DB_PATH)
        sim.executar_bateria_testes(NIVEIS_CONHECIMENTO, ERRO_FIXO, N_SIMULACOES)
    else:
        print("Erro: Nenhum dado gerado.")