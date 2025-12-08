import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import warnings

# Configurações visuais
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (16, 7)
warnings.filterwarnings("ignore")

def carregar_dados(db_path):
    try:
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
        WHERE g.resposta != 'X' 
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Erro ao conectar no banco: {e}")
        return pd.DataFrame()

def classificar_alternativas(df):
    df_me = df[df['tipo_prova'] == 'MULTIPLA_ESCOLHA'].copy()
    if df_me.empty:
        df['qtd_alternativas'] = 0
        return df

    provas_com_e = df_me[df_me['resposta'] == 'E'][['concurso', 'cargo']].drop_duplicates()
    provas_com_e['chave'] = provas_com_e['concurso'] + "_" + provas_com_e['cargo']
    chaves_5_itens = set(provas_com_e['chave'])
    
    def verificar_qtd(row):
        if row['tipo_prova'] != 'MULTIPLA_ESCOLHA': return 0
        chave = row['concurso'] + "_" + row['cargo']
        return 5 if chave in chaves_5_itens else 4

    df['qtd_alternativas'] = df.apply(verificar_qtd, axis=1)
    return df

def calcular_distribuicoes(df):
    """Calcula a PORCENTAGEM de cada letra por prova"""
    contagem = df.groupby(['concurso', 'cargo', 'tipo_prova', 'qtd_alternativas'])['resposta'].value_counts(normalize=True).unstack(fill_value=0)
    contagem = contagem * 100
    return contagem.reset_index()

def calcular_contagens_absolutas(df):
    """Calcula a QUANTIDADE REAL (número inteiro) de cada letra por prova"""
    contagem = df.groupby(['concurso', 'cargo', 'tipo_prova', 'qtd_alternativas'])['resposta'].value_counts(normalize=False).unstack(fill_value=0)
    return contagem.reset_index()

def analisar_estatisticas(series_pct, series_qtd, nome_analise, equilibrio_teorico):
    if series_pct.empty: return 0
    
    # Estatísticas de Tendência Central e Dispersão
    media = series_pct.mean()
    p10, p25, p50, p75, p90 = series_pct.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    
    # Total Absoluto
    total_itens = series_qtd.sum() if not series_qtd.empty else 0
    
    print(f"\n--- {nome_analise} ---")
    print(f"Total de Questões Analisadas: {int(total_itens)}")
    print(f"Média Real: {media:.2f}% (Teórico: {equilibrio_teorico}%)")
    print(f"Quartis: 1º(25%): {p25:.2f}% | Mediana(50%): {p50:.2f}% | 3º(75%): {p75:.2f}%")
    print(f"Intervalo 80% (P10-P90): {p10:.2f}% a {p90:.2f}%")
    
    return media

def plotar_certo_errado(df_ce_pct, df_ce_qtd):
    colunas_disponiveis = [col for col in ['C', 'E'] if col in df_ce_pct.columns]
    if not colunas_disponiveis: return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    if len(colunas_disponiveis) == 1: axes = [axes]
    
    print("\n=== ESTATÍSTICAS: CERTO vs ERRADO ===")

    configs = {
        'C': {'cor': 'green', 'titulo': 'CERTO (C)'},
        'E': {'cor': '#d62728', 'titulo': 'ERRADO (E)'}
    }

    for i, letra in enumerate(colunas_disponiveis):
        ax = axes[i]
        dados_pct = df_ce_pct[letra]
        dados_qtd = df_ce_qtd[letra] if letra in df_ce_qtd.columns else pd.Series()
        
        config = configs[letra]
        
        # Histograma
        sns.histplot(dados_pct, kde=True, bins=20, color=config['cor'], stat="density", ax=ax)
        
        media = analisar_estatisticas(dados_pct, dados_qtd, f"Gabarito {letra}", 50)
        
        # Linhas Verticais
        ax.axvline(media, color='black', linestyle='--', linewidth=2, label=f'Média Real: {media:.1f}%')
        ax.axvline(50, color='blue', linestyle='-', linewidth=2.5, label='Teórico (50%)')
        
        # Ajuste Visual
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
        ax.set_xlim(35, 65)
        
        ax.set_title(f'Distribuição - {config["titulo"]}', fontsize=14)
        ax.set_xlabel(f'% de itens {letra}')
        if i == 0: ax.set_ylabel('Densidade')
        ax.legend(loc='upper right')

    plt.suptitle('Balanceamento Cebraspe: CERTO vs ERRADO', fontsize=16)
    plt.tight_layout()
    plt.show()

def plotar_multipla_escolha(df_me_pct, df_me_qtd, n_alternativas):
    if n_alternativas == 5:
        letras = ['A', 'B', 'C', 'D', 'E']
        equilibrio = 20.0
        titulo = "Provas de 5 Alternativas (A-E)"
        limite_min, limite_max = 5, 35 
    else:
        letras = ['A', 'B', 'C', 'D']
        equilibrio = 25.0
        titulo = "Provas de 4 Alternativas (A-D)"
        limite_min, limite_max = 10, 40 

    letras_presentes = [l for l in letras if l in df_me_pct.columns]
    if not letras_presentes: return

    fig, axes = plt.subplots(1, len(letras_presentes), figsize=(3 * len(letras_presentes), 6), sharey=True)
    if len(letras_presentes) == 1: axes = [axes]

    print(f"\n=== ESTATÍSTICAS: {titulo} ===")

    for i, letra in enumerate(letras_presentes):
        ax = axes[i]
        dados_pct = df_me_pct[letra]
        dados_qtd = df_me_qtd[letra] if letra in df_me_qtd.columns else pd.Series()
        
        sns.histplot(dados_pct, kde=True, ax=ax, color=sns.color_palette("husl", 5)[i], stat="density")
        media = analisar_estatisticas(dados_pct, dados_qtd, f"Letra {letra}", equilibrio)
        
        ax.axvline(media, color='black', linestyle='--', linewidth=1.5, label=f'Real: {media:.1f}%')
        ax.axvline(equilibrio, color='blue', linestyle='-', linewidth=2, label=f'Teórico ({equilibrio:.0f}%)')
        
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))
        ax.set_xlim(limite_min, limite_max)
        
        ax.set_title(f'Letra {letra}', fontsize=12)
        ax.set_xlabel('% na prova')
        if i == 0: ax.set_ylabel('Densidade')
        ax.legend(fontsize='small')

    plt.suptitle(f'Distribuição de Gabaritos - {titulo}', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # AJUSTE O CAMINHO AQUI
    CAMINHO_DB = "../dada-scrapping/concursos_data.db"
    
    print(f"Lendo banco de dados em: {CAMINHO_DB}...")
    df_bruto = carregar_dados(CAMINHO_DB)
    
    if not df_bruto.empty:
        print("Classificando tipos de prova...")
        df_classificado = classificar_alternativas(df_bruto)
        
        # Calcula Percentagens
        df_dist = calcular_distribuicoes(df_classificado)
        # Calcula Quantidades Absolutas
        df_counts = calcular_contagens_absolutas(df_classificado)
        
        # Filtros de Percentagem
        df_ce = df_dist[df_dist['tipo_prova'] == 'CERTO_ERRADO']
        df_me_5 = df_dist[(df_dist['tipo_prova'] == 'MULTIPLA_ESCOLHA') & (df_dist['qtd_alternativas'] == 5)]
        df_me_4 = df_dist[(df_dist['tipo_prova'] == 'MULTIPLA_ESCOLHA') & (df_dist['qtd_alternativas'] == 4)]
        
        # Filtros de Quantidade
        df_counts_ce = df_counts[df_counts['tipo_prova'] == 'CERTO_ERRADO']
        df_counts_me_5 = df_counts[(df_counts['tipo_prova'] == 'MULTIPLA_ESCOLHA') & (df_counts['qtd_alternativas'] == 5)]
        df_counts_me_4 = df_counts[(df_counts['tipo_prova'] == 'MULTIPLA_ESCOLHA') & (df_counts['qtd_alternativas'] == 4)]
        
        if not df_ce.empty: 
            plotar_certo_errado(df_ce, df_counts_ce)
        if not df_me_5.empty: 
            plotar_multipla_escolha(df_me_5, df_counts_me_5, 5)
        if not df_me_4.empty: 
            plotar_multipla_escolha(df_me_4, df_counts_me_4, 4)
    else:
        print("Nenhum dado encontrado.")