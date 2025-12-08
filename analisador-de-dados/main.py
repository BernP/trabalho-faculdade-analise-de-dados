import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Configuração visual dos gráficos
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def carregar_dados(db_path="../dada-scrapping/concursos_data.db"):
    """Lê o SQLite e retorna um DataFrame Pandas"""
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        c.nome AS concurso,
        cg.nome_cargo AS cargo,
        cg.tipo_prova,
        g.resposta
    FROM gabaritos g
    JOIN cargos cg ON g.cargo_id = cg.id
    JOIN concursos c ON cg.concurso_id = c.id
    WHERE g.resposta != 'X'  -- Ignora anuladas para análise de balanceamento
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def calcular_distribuicoes(df):
    """Calcula a proporção de cada alternativa por prova (Cargo)"""
    # Agrupa por Concurso e Cargo e conta as respostas
    contagem = df.groupby(['concurso', 'cargo', 'tipo_prova'])['resposta'].value_counts(normalize=True).unstack(fill_value=0)
    
    # Transforma em porcentagem (0-100)
    contagem = contagem * 100
    return contagem.reset_index()

def analisar_estatisticas(series, nome_analise):
    """Calcula Média, Mediana, Moda e Intervalos"""
    media = series.mean()
    mediana = series.median()
    
    # Moda (arredondamos para 1 casa decimal para encontrar a moda real em dados contínuos)
    moda = series.round(1).mode()
    moda_str = ", ".join([f"{m:.1f}%" for m in moda]) if not moda.empty else "N/A"

    # Intervalos (Quantis)
    # 80% central = entre 10% e 90%
    p10, p90 = series.quantile([0.10, 0.90])
    # 90% central = entre 5% e 95%
    p05, p95 = series.quantile([0.05, 0.95])
    # 100% = Mínimo e Máximo
    minimo, maximo = series.min(), series.max()

    print(f"\n--- Estatísticas: {nome_analise} ---")
    print(f"Média:   {media:.2f}%")
    print(f"Mediana: {mediana:.2f}%")
    print(f"Moda (~): {moda_str}")
    print(f"Intervalo 80% (maioria): {p10:.2f}% a {p90:.2f}%")
    print(f"Intervalo 90% (amplo):   {p05:.2f}% a {p95:.2f}%")
    print(f"Amplitude Total (100%):  {minimo:.2f}% a {maximo:.2f}%")
    
    return media  # Retorna média para plotar linha no gráfico

def plotar_certo_errado(df_ce):
    """
    Gráfico 1: Distribuição da razão Certo/Errado
    Focaremos na porcentagem de 'C' (Certo). Se for 50%, está equilibrado.
    """
    if 'C' not in df_ce.columns:
        print("Dados insuficientes para análise Certo/Errado.")
        return

    plt.figure()
    sns.histplot(df_ce['C'], kde=True, bins=20, color='green')
    
    media = analisar_estatisticas(df_ce['C'], "Proporção de 'Certo' em provas C/E")
    
    plt.axvline(media, color='red', linestyle='--', label=f'Média: {media:.1f}%')
    plt.axvline(50, color='blue', linestyle=':', label='Equilíbrio Perfeito (50%)')
    
    plt.title('Distribuição da Porcentagem de Itens "CERTO" por Prova')
    plt.xlabel('Porcentagem de Gabaritos "C"')
    plt.ylabel('Frequência (Quantidade de Provas)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plotar_multipla_escolha(df_me):
    """
    5 Gráficos: Distribuição de A, B, C, D, E
    """
    alternativas = ['A', 'B', 'C', 'D', 'E']
    colunas_presentes = [letra for letra in alternativas if letra in df_me.columns]
    
    if not colunas_presentes:
        print("Nenhuma prova de Múltipla Escolha encontrada.")
        return

    # Cria uma figura com 5 subplots (ou menos se não tiver todas as letras)
    fig, axes = plt.subplots(1, len(colunas_presentes), figsize=(20, 5), sharey=True)
    
    if len(colunas_presentes) == 1: axes = [axes] # Garante que seja lista se só tiver 1

    for i, letra in enumerate(colunas_presentes):
        ax = axes[i]
        dados = df_me[letra]
        
        # Histograma
        sns.histplot(dados, kde=True, ax=ax, color=sns.color_palette("husl", 5)[i])
        
        # Estatísticas no Console
        media = analisar_estatisticas(dados, f"Alternativa {letra} (Múltipla Escolha)")
        
        # Linhas de referência no gráfico
        ax.axvline(media, color='black', linestyle='--', label=f'Média: {media:.1f}%')
        ax.axvline(20, color='gray', linestyle=':', label='Teórico (20%)') # Em 5 alternativas, 20% é o equilíbrio
        
        ax.set_title(f'Distribuição da Letra {letra}')
        ax.set_xlabel(f'% de {letra} na prova')
        if i == 0: ax.set_ylabel('Frequência')
        ax.legend()

    plt.suptitle('Análise de Balanceamento de Alternativas (Múltipla Escolha)', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Carregando dados do banco...")
    df_bruto = carregar_dados()
    
    if df_bruto.empty:
        print("O banco de dados está vazio ou sem gabaritos válidos.")
    else:
        # Calcula as % de cada prova
        df_distribuicao = calcular_distribuicoes(df_bruto)
        
        # Separa os dataframes por tipo
        df_ce = df_distribuicao[df_distribuicao['tipo_prova'] == 'CERTO_ERRADO']
        df_me = df_distribuicao[df_distribuicao['tipo_prova'] == 'MULTIPLA_ESCOLHA']

        print(f"\nProvas Certo/Errado analisadas: {len(df_ce)}")
        print(f"Provas Múltipla Escolha analisadas: {len(df_me)}")

        # 1. Análise Certo/Errado
        if not df_ce.empty:
            plotar_certo_errado(df_ce)
        
        # 2. Análise Múltipla Escolha
        if not df_me.empty:
            plotar_multipla_escolha(df_me)