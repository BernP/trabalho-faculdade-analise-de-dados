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
import math

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
# 2. GERADOR DE DADOS (ETL & BASE)
# ==========================================
class GeradorDeDados:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def _obter_todas_provas(self):
        query = "SELECT DISTINCT c.nome, cg.nome_cargo, cg.id, cg.tipo_prova FROM cargos cg JOIN concursos c ON cg.concurso_id = c.id"
        return pd.read_sql_query(query, self.conn)

    def _carregar_gabarito(self, cargo_id):
        query = "SELECT resposta FROM gabaritos WHERE cargo_id = ? AND resposta != 'X' ORDER BY numero_questao"
        cursor = self.conn.cursor()
        cursor.execute(query, (cargo_id,))
        return [row[0] for row in cursor.fetchall()]

    def _determinar_opcoes(self, tipo_prova, gabarito):
        if tipo_prova == 'CERTO_ERRADO': return ['C', 'E'], "CERTO_ERRADO"
        else: return (['A', 'B', 'C', 'D', 'E'], "MULTIPLA_5") if 'E' in gabarito else (['A', 'B', 'C', 'D'], "MULTIPLA_4")

    def _gerar_cenario(self, gabarito_real, opcoes, conhecimento, erro):
        total = len(gabarito_real)
        n_tentativas = int(total * conhecimento)
        indices_tentativa = np.random.choice(total, n_tentativas, replace=False)
        n_erros = int(n_tentativas * erro)
        indices_erros = np.random.choice(indices_tentativa, n_erros, replace=False)
        set_erros = set(indices_erros)
        set_tentativas = set(indices_tentativa)
        
        folha = [None] * total
        acertos_ini = 0; erros_ini = 0
        
        for i in range(total):
            if i in set_tentativas:
                if i in set_erros:
                    erradas = [op for op in opcoes if op != gabarito_real[i]]
                    folha[i] = random.choice(erradas); erros_ini += 1
                else: folha[i] = gabarito_real[i]; acertos_ini += 1
        return folha, acertos_ini, erros_ini

    def _calcular_nota_base(self, acertos_ini, erros_ini, tipo_grupo):
        return (acertos_ini - erros_ini) if tipo_grupo == "CERTO_ERRADO" else acertos_ini

    def _calcular_nota(self, gabarito_cand, gabarito_real, tipo_grupo):
        acertos = 0; erros = 0
        for c, r in zip(gabarito_cand, gabarito_real):
            if c == r: acertos += 1
            else: erros += 1
        return (acertos - erros) if tipo_grupo == "CERTO_ERRADO" else acertos

    def gerar_dataset_completo(self, lista_conhecimento, lista_erro, n_simulacoes=1000):
        df_provas = self._obter_todas_provas()
        resultados = []
        total_provas = len(df_provas)
        print(f"=== INICIANDO SIMULAÇÃO MASSIVA (COMPLETA) ===")
        print(f"Provas: {total_provas} | Simulações/Cenário: {n_simulacoes}")

        for idx, row in df_provas.iterrows():
            gabarito_real = self._carregar_gabarito(row['id'])
            if not gabarito_real: continue
            opcoes, grupo = self._determinar_opcoes(row['tipo_prova'], gabarito_real)
            
            for k in lista_conhecimento:
                for e in lista_erro:
                    lista_eficiencias = [] 
                    lista_ganho_pct = []
                    acima_50 = 0
                    
                    for _ in range(n_simulacoes):
                        parcial, acertos_ini, erros_ini = self._gerar_cenario(gabarito_real, opcoes, k, e)
                        indices_chute = [i for i, x in enumerate(parcial) if x is None]
                        
                        nota_base = self._calcular_nota_base(acertos_ini, erros_ini, grupo)
                        
                        final, _ = EstrategiaChute.menos_marcada(parcial, opcoes)
                        nota_tec = self._calcular_nota(final, gabarito_real, grupo)
                        
                        # Cálculo de Eficiência (Acertos / Chutes)
                        if indices_chute:
                            acertos_chute = sum(1 for i in indices_chute if final[i] == gabarito_real[i])
                            eficiencia = acertos_chute / len(indices_chute)
                            
                            # Cálculo de Ganho Percentual (Pontos Extras / Chutes)
                            # C/E: (Acertos - Erros) / Chutes
                            erros_chute = len(indices_chute) - acertos_chute
                            saldo_chute = (acertos_chute - erros_chute) if grupo == "CERTO_ERRADO" else acertos_chute
                            ganho_p = saldo_chute / len(indices_chute)
                        else:
                            eficiencia = 0.0
                            ganho_p = 0.0
                        
                        lista_eficiencias.append(eficiencia)
                        lista_ganho_pct.append(ganho_p)
                        if eficiencia >= 0.50: acima_50 += 1

                    # Estatísticas REAIS da Amostra
                    arr_efi = np.array(lista_eficiencias)
                    arr_ganho = np.array(lista_ganho_pct)
                    
                    resultados.append({
                        'Grupo': grupo,
                        'Conhecimento': k,
                        'Erro': e,
                        # Métricas de Eficiência (% Acerto)
                        'Eficiencia_Media': np.mean(arr_efi),
                        'Eficiencia_Mediana': np.median(arr_efi),
                        'Eficiencia_Min': np.min(arr_efi),
                        'Eficiencia_Max': np.max(arr_efi),
                        'Eficiencia_Q1': np.percentile(arr_efi, 25),
                        'Eficiencia_Q3': np.percentile(arr_efi, 75),
                        # Métricas de Ganho Relativo (% Pontos/Chute)
                        'GanhoPct_Media': np.mean(arr_ganho),
                        'Prob_Acima_50': acima_50 / n_simulacoes
                    })
            
            if (idx + 1) % 10 == 0: print(f"   Processado {idx + 1}/{total_provas}...")

        return pd.DataFrame(resultados)

# ==========================================
# 3. ANALISADOR (GRÁFICOS E TABELAS)
# ==========================================
class AnalisadorEstatistico:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def _config_grafico(self, tit, xl, yl):
        plt.title(tit, fontsize=14)
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.legend(title='Taxa de Erro')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

    # --- 1. Gráfico de Linha: Eficiência Média (% Acerto) ---
    def plotar_curvas_eficiencia_media(self):
        print("\n[GRÁFICOS] Gerando curvas de Eficiência Média (Linha)...")
        grupos = self.df['Grupo'].unique()
        for g in grupos:
            df_g = self.df[self.df['Grupo'] == g]
            if df_g.empty: continue
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df_g, x='Conhecimento', y='Eficiencia_Media', hue='Erro', palette='magma', marker='o')
            
            base = 0.5 if g == 'CERTO_ERRADO' else (0.2 if '5' in g else 0.25)
            plt.axhline(base, color='blue', linestyle='--', label=f'Aleatório ({base:.0%})')
            self._config_grafico(f'Evolução da Eficiência do Chute - {g}', 'Nível de Conhecimento', 'Taxa Média de Acerto (%)')
            plt.show()

    # --- 2. Gráfico de Linha: Ganho Percentual (% Pontos) ---
    def plotar_curvas_ganho_percentual(self):
        print("\n[GRÁFICOS] Gerando curvas de Ganho Percentual (Linha)...")
        grupos = self.df['Grupo'].unique()
        for g in grupos:
            df_g = self.df[self.df['Grupo'] == g]
            if df_g.empty: continue
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=df_g, x='Conhecimento', y='GanhoPct_Media', hue='Erro', palette='viridis', marker='o')
            
            plt.axhline(0, color='red', linestyle='--', label='Zero (Neutro)')
            self._config_grafico(f'Rendimento Líquido do Chute - {g}', 'Nível de Conhecimento', 'Ganho (% Pontos sobre Chutes)')
            plt.show()

    # --- 3. Gráfico de Dispersão: Correlação ---
    def plotar_correlacao(self):
        print("\n[GRÁFICOS] Gerando Correlação (Scatter)...")
        grupos = self.df['Grupo'].unique()
        for g in grupos:
            df_g = self.df[self.df['Grupo'] == g]
            if df_g.empty: continue
            plt.figure(figsize=(10, 6))
            sns.regplot(data=df_g, x='Conhecimento', y='Eficiencia_Media', scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
            self._config_grafico(f'Correlação: Conhecimento vs Eficiência - {g}', 'Conhecimento', 'Eficiência Média')
            plt.show()

    # --- 4. Gráfico de Sino: Distribuição da Eficiência ---
    def plotar_distribuicao_sino_eficiencia(self, erro_alvo=0.10):
        print(f"\n[GRÁFICOS] Gerando Curvas de Sino (Eficiência) para Erro {erro_alvo*100:.0f}%...")
        df_f = self.df[np.isclose(self.df['Erro'], erro_alvo)]
        for g in df_f['Grupo'].unique():
            df_g = df_f[df_f['Grupo'] == g]
            plt.figure(figsize=(12, 6))
            # Usa Eficiencia_Media de cada prova como ponto de dados
            sns.histplot(data=df_g, x='Eficiencia_Media', hue='Conhecimento', kde=True, element="step", palette="viridis", stat="density", common_norm=False)
            
            plt.title(f'Distribuição da Eficiência Média - {g} (Erro {erro_alvo:.0%})', fontsize=14)
            plt.xlabel('Taxa de Acerto no Chute (%)')
            plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            
            base = 0.5 if g == 'CERTO_ERRADO' else (0.2 if '5' in g else 0.25)
            plt.axvline(base, color='red', linestyle='--', label=f'Aleatório ({base:.0%})')
            plt.legend()
            plt.show()

    # --- 5. Tabela Rica ---
    def imprimir_tabela_definitiva(self, erro_alvo=0.10):
        print("\n" + "="*160)
        print(f" TABELA DEFINITIVA: EFICIÊNCIA DO CHUTE (Acertos / Tentativas) - Erro {erro_alvo*100:.0f}% ".center(160))
        print("="*160)
        
        df_f = self.df[np.isclose(self.df['Erro'], erro_alvo)]
        
        for g in df_f['Grupo'].unique():
            print(f"\n>>> GRUPO: {g}")
            print("-" * 160)
            print(f"{'Conhec.':<8} | {'Média':<8} {'Mediana':<8} {'Min(Geral)':<10} {'Min(Méd)':<8} {'Q1(Méd)':<8} {'Q3(Méd)':<8} {'Max(Méd)':<8} {'Max(Geral)':<10} | {'Prob >= 50%':<12}")
            print("-" * 160)
            
            for n in sorted(df_f['Conhecimento'].unique()):
                d = df_f[(df_f['Grupo']==g) & (df_f['Conhecimento']==n)]
                if d.empty: continue
                
                # Agregação das provas
                media_das_medias = d['Eficiencia_Media'].mean()
                mediana_das_medianas = d['Eficiencia_Mediana'].median()
                
                # Extremos Globais (resgatados das colunas Min/Max salvas no CSV)
                min_global = d['Eficiencia_Min'].min()
                max_global = d['Eficiencia_Max'].max()
                
                # Distribuição das Médias
                min_media = d['Eficiencia_Media'].min()
                q1_media = d['Eficiencia_Media'].quantile(0.25)
                q3_media = d['Eficiencia_Media'].quantile(0.75)
                max_media = d['Eficiencia_Media'].max()
                
                prob_50 = d['Prob_Acima_50'].mean()
                
                print(f"{n*100:.0f}%     | {media_das_medias:.2%}   {mediana_das_medianas:.2%}   {min_global:+.2%}     {min_media:.2%}   {q1_media:.2%}   {q3_media:.2%}   {max_media:.2%}   {max_global:+.2%}     | {prob_50:.2%}")
            print("-" * 160)

# ==========================================
# 4. EXECUÇÃO
# ==========================================
if __name__ == "__main__":
    DB_PATH = "../dada-scrapping/concursos_data.db"
    CSV_PATH = "dados_simulacao_v7_completa.csv"
    
    # ATENÇÃO: Deixe True na primeira vez para criar o CSV novo com as colunas Min/Max/Q
    RODAR_NOVA_SIMULACAO = True 
    
    if RODAR_NOVA_SIMULACAO:
        gerador = GeradorDeDados(DB_PATH)
        # Roda 1000 simulações para cada cenário e salva as estatísticas detalhadas
        df = gerador.gerar_dataset_completo([0.5, 0.6, 0.7, 0.8, 0.9], [0.05, 0.1, 0.2], 1000)
        df.to_csv(CSV_PATH, index=False)
        print(f"Dados salvos em {CSV_PATH}")
    
    if os.path.exists(CSV_PATH):
        ana = AnalisadorEstatistico(CSV_PATH)
        
        # 1. Gráficos de Linha (Tendência) - RESTAURADOS
        ana.plotar_curvas_eficiencia_media()
        ana.plotar_curvas_ganho_percentual()
        
        # 2. Gráfico de Dispersão (Scatter) - RESTAURADO
        ana.plotar_correlacao()
        
        # 3. Gráfico de Sino (Distribuição)
        ana.plotar_distribuicao_sino_eficiencia(erro_alvo=0.10)
        
        # 4. Tabela Rica
        ana.imprimir_tabela_definitiva(erro_alvo=0.10)
    else:
        print("CSV não encontrado.")