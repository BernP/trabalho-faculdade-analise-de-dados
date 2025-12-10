import sqlite3
import pandas as pd
import numpy as np
import random
from scipy import stats
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import os

# Configurações
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ==========================================
# 1. LÓGICA DE NEGÓCIO (REGRA DE CHUTE)
# ==========================================
class EstrategiaChute:
    @staticmethod
    def menos_marcada(gabarito_parcial, opcoes_possiveis):
        marcados = [x for x in gabarito_parcial if x is not None]
        if not marcados:
            return [random.choice(opcoes_possiveis) for _ in gabarito_parcial], None
            
        contagem = {op: marcados.count(op) for op in opcoes_possiveis}
        min_valor = min(contagem.values())
        candidatas = [k for k, v in contagem.items() if v == min_valor]
        letra_escolhida = random.choice(candidatas)
        gabarito_final = [x if x is not None else letra_escolhida for x in gabarito_parcial]
        return gabarito_final, letra_escolhida

# ==========================================
# 2. GERADOR DE DADOS (ETL)
# ==========================================
class GeradorDeDados:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

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

    def _gerar_cenario(self, gabarito_real, opcoes, conhecimento, erro):
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
        return (acertos - erros) if tipo_grupo == "CERTO_ERRADO" else acertos

    def _calcular_nota_base(self, acertos_ini, erros_ini, tipo_grupo):
        if tipo_grupo == "CERTO_ERRADO":
            return acertos_ini - erros_ini
        return acertos_ini

    def _calcular_aleatoria_esperada(self, total, acertos_ini, erros_ini, grupo, n_opcoes):
        n_chutes = total - (acertos_ini + erros_ini)
        nota_base = self._calcular_nota_base(acertos_ini, erros_ini, grupo)
        if grupo == "CERTO_ERRADO": return nota_base
        return nota_base + (n_chutes * (1 / n_opcoes))

    def gerar_dataset_completo(self, lista_conhecimento, lista_erro, n_simulacoes=1000):
        df_provas = self._obter_todas_provas()
        resultados = []
        total_provas = len(df_provas)
        
        start_time = time.time()
        print(f"=== INICIANDO SIMULAÇÃO MASSIVA ===")
        print(f"Provas: {total_provas} | Simulações/Cenário: {n_simulacoes}")

        for idx, row in df_provas.iterrows():
            gabarito_real = self._carregar_gabarito(row['id'])
            if not gabarito_real: continue
            
            opcoes, grupo = self._determinar_opcoes(row['tipo_prova'], gabarito_real)
            
            for k in lista_conhecimento:
                for e in lista_erro:
                    soma_tec = 0
                    soma_rnd = 0
                    soma_base = 0
                    soma_eficiencia = 0
                    soma_n_chutes = 0
                    
                    for _ in range(n_simulacoes):
                        parcial, acertos_ini, erros_ini = self._gerar_cenario(gabarito_real, opcoes, k, e)
                        
                        # Calcula quantos chutes foram dados (Questões em branco)
                        indices_chute = [i for i, x in enumerate(parcial) if x is None]
                        n_chutes_atual = len(indices_chute)
                        soma_n_chutes += n_chutes_atual

                        nota_base = self._calcular_nota_base(acertos_ini, erros_ini, grupo)
                        
                        final, _ = EstrategiaChute.menos_marcada(parcial, opcoes)
                        nota_tec = self._calcular_nota(final, gabarito_real, grupo)
                        nota_rnd = self._calcular_aleatoria_esperada(len(gabarito_real), acertos_ini, erros_ini, grupo, len(opcoes))
                        
                        if n_chutes_atual > 0:
                            acertos_chute = sum(1 for i in indices_chute if final[i] == gabarito_real[i])
                            eficiencia = acertos_chute / n_chutes_atual
                        else:
                            eficiencia = 0
                        soma_eficiencia += eficiencia

                        soma_tec += nota_tec
                        soma_rnd += nota_rnd
                        soma_base += nota_base

                    # Médias
                    media_tec = soma_tec / n_simulacoes
                    media_base = soma_base / n_simulacoes
                    media_chutes = soma_n_chutes / n_simulacoes
                    ganho_bruto_medio = media_tec - media_base
                    
                    # Ganho Percentual = (Pontos Ganhos / Quantidade de Chutes)
                    # Evita divisão por zero
                    ganho_pct = (ganho_bruto_medio / media_chutes) if media_chutes > 0 else 0.0

                    resultados.append({
                        'Concurso': row['nome'],
                        'Cargo': row['nome_cargo'],
                        'Grupo': grupo,
                        'Conhecimento': k,
                        'Erro': e,
                        'Media_Tecnica': media_tec,
                        'Media_Aleatoria': soma_rnd / n_simulacoes,
                        'Ganho_Bruto': ganho_bruto_medio,
                        'Ganho_Percentual': ganho_pct, # NOVA COLUNA RELATIVA
                        'Eficiencia_Media': soma_eficiencia / n_simulacoes
                    })
            
            if (idx + 1) % 10 == 0:
                print(f"   Processado {idx + 1}/{total_provas} provas...")

        return pd.DataFrame(resultados)

# ==========================================
# 3. ANALISADOR (VISUALIZAÇÃO E RELATÓRIOS)
# ==========================================
class AnalisadorEstatistico:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        print(f"\nDados carregados: {len(self.df)} registros.")

    def _configurar_grafico(self, title, xlabel, ylabel, is_percent_x=True, is_percent_y=False):
        plt.title(title, fontsize=14)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if is_percent_x: plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        if is_percent_y: plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.legend(title='Taxa de Erro')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

    def plotar_ganho_bruto(self):
        grupos = self.df['Grupo'].unique()
        print("\n[GRÁFICOS] Gerando curvas de Ganho de Pontos (Bruto)...")
        for grupo in grupos:
            df_g = self.df[self.df['Grupo'] == grupo]
            if df_g.empty: continue
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df_g, x='Conhecimento', y='Ganho_Bruto', hue='Erro', palette='viridis', marker='o')
            plt.axhline(0, color='red', linestyle='--', linewidth=1)
            self._configurar_grafico(f'Pontos Ganhos com o Chute: {grupo}', 'Nível de Conhecimento', 'Pontos Extras (vs Em Branco)')
            plt.show()

    def plotar_eficiencia(self):
        grupos = self.df['Grupo'].unique()
        print("\n[GRÁFICOS] Gerando curvas de Eficiência (% Acerto)...")
        for grupo in grupos:
            df_g = self.df[self.df['Grupo'] == grupo]
            if df_g.empty: continue
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df_g, x='Conhecimento', y='Eficiencia_Media', hue='Erro', palette='magma', marker='o')
            chance_aleatoria = 0.5 if grupo == 'CERTO_ERRADO' else (0.2 if grupo == 'MULTIPLA_5' else 0.25)
            plt.axhline(chance_aleatoria, color='blue', linestyle='--', label=f'Aleatório ({chance_aleatoria:.0%})')
            self._configurar_grafico(f'Qualidade do Chute: {grupo}', 'Nível de Conhecimento', 'Taxa de Acerto (%)', is_percent_y=True)
            plt.show()

    def plotar_correlacao_conhecimento_eficacia(self):
        print("\n[GRÁFICOS] Gerando correlação Conhecimento x Eficiência...")
        grupos = self.df['Grupo'].unique()
        for grupo in grupos:
            df_g = self.df[self.df['Grupo'] == grupo]
            if df_g.empty: continue
            plt.figure(figsize=(10, 6))
            sns.regplot(data=df_g, x='Conhecimento', y='Eficiencia_Media', scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
            self._configurar_grafico(f'Correlação: Quanto mais sei, melhor chuto? ({grupo})', 'Nível de Conhecimento', 'Eficiência do Chute', is_percent_y=True)
            plt.show()

    def plotar_distribuicao_ganhos(self, erro_alvo=0.10):
        print(f"\n[GRÁFICOS] Gerando Curvas de Distribuição (Sino) para Erro {erro_alvo*100:.0f}%...")
        df_filtro = self.df[np.isclose(self.df['Erro'], erro_alvo)]
        grupos = df_filtro['Grupo'].unique()
        for grupo in grupos:
            df_g = df_filtro[df_filtro['Grupo'] == grupo]
            if df_g.empty: continue
            plt.figure(figsize=(12, 6))
            sns.histplot(data=df_g, x='Ganho_Bruto', hue='Conhecimento', kde=True, element="step", palette="viridis", stat="density", common_norm=False)
            plt.title(f'Distribuição dos Ganhos - {grupo} (Erro {erro_alvo*100:.0f}%)', fontsize=14)
            plt.xlabel('Ganho Médio de Pontos na Prova')
            plt.ylabel('Densidade')
            plt.axvline(0, color='red', linestyle='--', linewidth=2)
            plt.legend(title='Conhecimento')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()

    def imprimir_tabela_ganhos(self, erro_alvo=0.10):
        """
        Tabela descritiva expandida:
        Inclui estatísticas de Pontos Brutos (Total) e Ganho Percentual (Relativo).
        """
        print("\n" + "="*145)
        print(f" TABELA ESTATÍSTICA DETALHADA: GANHO BRUTO E RENDIMENTO DO CHUTE (Erro fixado em {erro_alvo*100:.0f}%) ".center(145))
        print("="*145)
        
        df_filtro = self.df[np.isclose(self.df['Erro'], erro_alvo)]
        grupos = df_filtro['Grupo'].unique()
        niveis = sorted(df_filtro['Conhecimento'].unique())
        
        for grupo in grupos:
            print(f"\n>>> GRUPO: {grupo}")
            # Cabeçalho Duplo
            print("-" * 145)
            print(f"{' ':10} | {'[ PONTUAÇÃO BRUTA (Pontos) ]':^58} | {'[ RENDIMENTO % SOBRE O CHUTE (Relativo) ]':^45}")
            print(f"{'Conhec.':<10} | {'Média':<8} {'Mediana':<8} {'Moda':<8} {'Min':<8} {'Q1':<8} {'Q3':<8} {'Max':<8} | {'Min %':<8} {'Q1 %':<8} {'Q3 %':<8} {'Max %':<8}")
            print("-" * 145)
            
            for nivel in niveis:
                df_nivel = df_filtro[(df_filtro['Grupo'] == grupo) & (df_filtro['Conhecimento'] == nivel)]
                dados_bruto = df_nivel['Ganho_Bruto']
                dados_pct = df_nivel['Ganho_Percentual']
                
                if dados_bruto.empty: continue
                
                # Stats Bruto
                media = dados_bruto.mean()
                mediana = dados_bruto.median()
                moda = dados_bruto.round(2).mode()[0] if not dados_bruto.round(2).mode().empty else 0
                min_b = dados_bruto.min()
                q1_b = dados_bruto.quantile(0.25)
                q3_b = dados_bruto.quantile(0.75)
                max_b = dados_bruto.max()
                
                # Stats Percentual (Relativo)
                min_p = dados_pct.min()
                q1_p = dados_pct.quantile(0.25)
                q3_p = dados_pct.quantile(0.75)
                max_p = dados_pct.max()
                
                print(f"{nivel*100:.0f}%       | {media:+.2f}    {mediana:+.2f}    {moda:+.2f}    {min_b:+.2f}    {q1_b:+.2f}    {q3_b:+.2f}    {max_b:+.2f}    | {min_p:+.2%}   {q1_p:+.2%}   {q3_p:+.2%}   {max_p:+.2%}")
            print("-" * 145)

    def relatorio_teste_hipotese_didatico(self, erro_alvo=0.10):
        print("\n" + "#"*80)
        print(f" RELATÓRIO CIENTÍFICO DE VALIDAÇÃO (Erro {erro_alvo*100:.0f}%) ".center(80))
        print("#"*80)
        
        print("\n1. DEFINIÇÃO")
        print("   Teste T Pareado comparando (Nota Técnica) vs (Nota Aleatória).")
        print("   Validamos se a técnica gera vantagem estatística real.")

        df_filtro = self.df[np.isclose(self.df['Erro'], erro_alvo)]
        niveis = sorted(df_filtro['Conhecimento'].unique())
        
        for nivel in niveis:
            print(f"\n" + "-"*60)
            print(f"   CENÁRIO: Candidato sabe {nivel*100:.0f}% da prova")
            print("-"*60)
            
            df_cenario = df_filtro[df_filtro['Conhecimento'] == nivel]
            grupos = sorted(df_cenario['Grupo'].unique())
            
            for grupo in grupos:
                df_g = df_cenario[df_cenario['Grupo'] == grupo]
                if len(df_g) < 2: continue
                
                stat, p_valor = stats.ttest_rel(df_g['Media_Tecnica'], df_g['Media_Aleatoria'], alternative='greater')
                vantagem = (df_g['Media_Tecnica'] - df_g['Media_Aleatoria']).mean()
                
                # Eficiência (% de acerto nos chutes)
                eficiencia = df_g['Eficiencia_Media'].mean() * 100
                base_aleatoria = 50.0 if 'CERTO' in grupo else (20.0 if '5' in grupo else 25.0)
                
                print(f"\n   > {grupo}:")
                print(f"     Vantagem sobre a Sorte: {vantagem:+.2f} pts")
                print(f"     Taxa Acerto Chute: {eficiencia:.2f}% (Base: {base_aleatoria:.0f}%)")
                print(f"     P-valor: {p_valor:.4e}")
                
                if p_valor < 0.05:
                    print("     [APROVADO] Técnica estatisticamente SUPERIOR.")
                else:
                    print("     [REPROVADO] Técnica não supera a sorte.")

# ==========================================
# 4. EXECUÇÃO
# ==========================================
if __name__ == "__main__":
    DB_PATH = "../dada-scrapping/concursos_data.db"
    CSV_PATH = "dados_simulacao_completa_v4.csv" # Nome novo para garantir
    
    # --- CONTROLE ---
    RODAR_NOVA_SIMULACAO = True # Mude para False para apenas analisar
    
    LISTA_CONHECIMENTO = [0.50, 0.60, 0.70, 0.80, 0.90]
    LISTA_ERRO = [0.05, 0.10, 0.20]
    SIMULACOES = 1000

    if RODAR_NOVA_SIMULACAO or not os.path.exists(CSV_PATH):
        gerador = GeradorDeDados(DB_PATH)
        df_completo = gerador.gerar_dataset_completo(LISTA_CONHECIMENTO, LISTA_ERRO, SIMULACOES)
        if not df_completo.empty:
            df_completo.to_csv(CSV_PATH, index=False)
            print(f"Simulação concluída. Dados salvos em {CSV_PATH}")
        else:
            print("Erro na simulação.")
            exit()
    
    if os.path.exists(CSV_PATH):
        analisador = AnalisadorEstatistico(CSV_PATH)
        analisador.plotar_ganho_bruto()
        analisador.plotar_eficiencia()
        analisador.plotar_correlacao_conhecimento_eficacia()
        analisador.imprimir_tabela_ganhos(erro_alvo=0.10)
        analisador.plotar_distribuicao_ganhos(erro_alvo=0.10)
        analisador.relatorio_teste_hipotese_didatico(erro_alvo=0.10)
    else:
        print("CSV não encontrado.")