import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import warnings
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chisquare
from scipy.stats import ttest_1samp

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

def executar_teste_z_balanceamento(df):
    """
    Testa se a proporção de CERTO é estatisticamente igual a 50%.
    """
    # Filtra apenas questões Certo/Errado
    df_ce = df[df['tipo_prova'] == 'CERTO_ERRADO']
    
    # Contagem de sucessos (Certo) e total de observações (n)
    n_sucessos = df_ce[df_ce['resposta'] == 'C'].shape[0]
    n_total = df_ce.shape[0]
    
    # Executa o teste bilateral (two-sided) comparando com 0.5
    stat, p_valor = proportions_ztest(count=n_sucessos, nobs=n_total, value=0.5)
    
    print(f"--- Teste 1: Z-Test para Proporção (Certo/Errado) ---")
    print(f"Proporção Observada: {(n_sucessos/n_total)*100:.2f}%")
    print(f"Estatística Z: {stat:.4f}")
    print(f"P-valor: {p_valor:.4e}")
    
    if p_valor < 0.05:
        print(">> Conclusão: Rejeita-se H0. O desvio de 50% NÃO é aleatório (há viés).")
    else:
        print(">> Conclusão: Aceita-se H0. O desvio é estatisticamente irrelevante.")

def executar_teste_qui_quadrado(df):
    """
    Verifica se a distribuição das 5 alternativas foge do padrão uniforme (20% cada).
    """
    # Filtra dados de 5 alternativas
    df_5 = df[(df['tipo_prova'] == 'MULTIPLA_ESCOLHA') & (df['qtd_alternativas'] == 5)]
    
    # Contagem observada
    contagem = df_5['resposta'].value_counts().sort_index()
    observado = contagem.values
    
    # Cálculo das esperadas (distribuição uniforme)
    total_questoes = observado.sum()
    esperado = [total_questoes / 5] * 5
    
    stat, p_valor = chisquare(f_obs=observado, f_exp=esperado)
    
    print(f"\n--- Teste 2: Qui-Quadrado (Distribuição Global A-E) ---")
    print(f"Qui-Quadrado: {stat:.4f}")
    print(f"P-valor: {p_valor:.4e}")
    
    if p_valor < 0.05:
        print(">> Conclusão: A distribuição global NÃO é uniforme. Há preferência por certas letras.")
    else:
        print(">> Conclusão: A distribuição segue o padrão uniforme esperado.")

def executar_teste_t_letra_a(df):
    """
    Testa se a média da Letra A é significativamente menor que 20%.
    """
    # 1. Prepara os dados: Calcula % de A para cada prova individualmente
    df_5 = df[(df['tipo_prova'] == 'MULTIPLA_ESCOLHA') & (df['qtd_alternativas'] == 5)]
    
    # Agrupa por prova e calcula %
    distribuicao_por_prova = df_5.groupby(['concurso', 'cargo'])['resposta'] \
                                 .value_counts(normalize=True).unstack(fill_value=0) * 100
    
    # Pega apenas a coluna 'A'
    amostra_a = distribuicao_por_prova['A']
    
    # 2. Executa o teste T (comparando com a média populacional 20)
    # alternative='less' verifica se é MENOR que 20
    stat, p_valor = ttest_1samp(amostra_a, popmean=20.0, alternative='less')
    
    print(f"\n--- Teste 3: Teste T (Viés Negativo da Letra A) ---")
    print(f"Média das Provas: {amostra_a.mean():.2f}%")
    print(f"Estatística T: {stat:.4f}")
    print(f"P-valor: {p_valor:.4e}")
    
    if p_valor < 0.05:
        print(">> Conclusão: Há evidência estatística de que a banca sub-representa a letra A.")
    else:
        print(">> Conclusão: A média baixa da letra A está dentro da margem de erro normal.")
def print_header(titulo):
    print("\n" + "="*70)
    print(f"{titulo.center(70)}")
    print("="*70)

def relatorio_z_test_detalhado(df):
    print_header("RELATÓRIO 1: TESTE Z (BALANCEAMENTO CERTO/ERRADO)")
    
    # 1. Preparação dos Dados
    df_ce = df[df['tipo_prova'] == 'CERTO_ERRADO']
    n_sucessos = df_ce[df_ce['resposta'] == 'C'].shape[0]
    n_total = df_ce.shape[0]
    prop_real = n_sucessos / n_total
    prop_teorica = 0.5
    
    print("PASSO 1: DEFINIÇÃO DAS HIPÓTESES")
    print("   H0 (Nula): A banca é justa. A proporção de 'Certo' é exatamente 50% (0.5).")
    print("   H1 (Alternativa): A banca tem viés. A proporção é diferente de 50%.")
    
    print("\nPASSO 2: COLETA DE DADOS")
    print(f"   Total de Questões (n): {n_total}")
    print(f"   Quantidade de 'Certo' (x): {n_sucessos}")
    print(f"   Proporção Observada (p̂): {prop_real:.4f} ({prop_real*100:.2f}%)")
    
    # 2. Execução do Teste
    stat, p_valor = proportions_ztest(count=n_sucessos, nobs=n_total, value=prop_teorica)
    
    print("\nPASSO 3: CÁLCULO ESTATÍSTICO (TESTE Z)")
    print(f"   Diferença encontrada: {(prop_real - prop_teorica)*100:.2f} pontos percentuais")
    print(f"   Z-Score calculado: {stat:.4f}")
    print("   (O Z-Score mede quantos 'desvios' estamos longe da média ideal de 0)")

    print("\nPASSO 4: INTERPRETAÇÃO DO P-VALOR")
    print(f"   P-valor: {p_valor:.4f} ({p_valor*100:.2f}%)")
    print("   Significado: É a chance de um resultado desse acontecer por pura sorte.")
    
    print("\nPASSO 5: VEREDITO FINAL")
    if p_valor < 0.05:
        print("   [!] REJEITA-SE H0.")
        print("   Conclusão: O desvio é estatisticamente significativo. A banca não é perfeitamente 50/50.")
    else:
        print("   [OK] ACEITA-SE H0.")
        print("   Conclusão: O desvio é estatisticamente irrelevante. Pode ser considerado aleatório.")


def relatorio_chi2_detalhado(df):
    print_header("RELATÓRIO 2: QUI-QUADRADO (DISTRIBUIÇÃO A-E)")
    
    # 1. Preparação
    df_5 = df[(df['tipo_prova'] == 'MULTIPLA_ESCOLHA') & (df['qtd_alternativas'] == 5)]
    contagem = df_5['resposta'].value_counts().sort_index()
    observado = contagem.values
    letras = contagem.index.tolist()
    
    if len(observado) != 5:
        print("Erro: Dados insuficientes para análise completa das 5 letras.")
        return

    total = observado.sum()
    esperado_por_letra = total / 5
    esperado = [esperado_por_letra] * 5
    
    print("PASSO 1: DEFINIÇÃO DAS HIPÓTESES")
    print("   H0: A distribuição é Uniforme (todas as letras têm a mesma chance).")
    print("   H1: A distribuição é Viciada (algumas letras aparecem mais que outras).")
    
    print("\nPASSO 2: COMPARATIVO OBSERVADO vs ESPERADO")
    print(f"   Total de questões: {total}. Esperado por letra: {esperado_por_letra:.1f}")
    print(f"   {'Letra':<5} | {'Real':<10} | {'Ideal':<10} | {'Diferença':<10}")
    print("   " + "-"*45)
    
    soma_erro_quadratico = 0
    for i in range(5):
        diff = observado[i] - esperado[i]
        erro_q = (diff ** 2) / esperado[i]
        soma_erro_quadratico += erro_q
        print(f"   {letras[i]:<5} | {observado[i]:<10} | {esperado[i]:<10.1f} | {diff:<+10.1f}")

    # 2. Execução
    stat, p_valor = chisquare(f_obs=observado)
    
    print("\nPASSO 3: CÁLCULO ESTATÍSTICO (QUI-QUADRADO)")
    print(f"   Chi2 Calculado (Soma dos erros normalizados): {stat:.4f}")
    print(f"   (Valor crítico para 5 alternativas é aprox 9.48. O seu deu {stat:.2f})")

    print("\nPASSO 4: VEREDITO FINAL")
    print(f"   P-valor: {p_valor:.8f}")
    
    if p_valor < 0.05:
        print("   [!] REJEITA-SE H0.")
        print("   Conclusão: A distribuição NÃO é uniforme. Há um padrão de preferência claro.")
    else:
        print("   [OK] ACEITA-SE H0.")
        print("   Conclusão: As diferenças são pequenas o suficiente para serem sorte.")


def relatorio_t_test_detalhado(df):
    print_header("RELATÓRIO 3: TESTE T (VIÉS DA LETRA A)")
    
    # 1. Preparação: Calcular % de A por prova
    df_5 = df[(df['tipo_prova'] == 'MULTIPLA_ESCOLHA') & (df['qtd_alternativas'] == 5)]
    distribuicao = df_5.groupby(['concurso', 'cargo'])['resposta'].value_counts(normalize=True).unstack(fill_value=0) * 100
    
    if 'A' not in distribuicao.columns: return
    amostra_a = distribuicao['A']
    
    media_obs = amostra_a.mean()
    media_teorica = 20.0
    desvio_padrao = amostra_a.std()
    n_provas = len(amostra_a)
    
    print("PASSO 1: DEFINIÇÃO DAS HIPÓTESES")
    print("   H0: A média da Letra A nas provas é igual a 20%.")
    print("   H1: A média da Letra A é significativamente MENOR que 20%.")
    
    print("\nPASSO 2: ANÁLISE DESCRITIVA DAS PROVAS")
    print(f"   Número de provas analisadas: {n_provas}")
    print(f"   Média observada da Letra A: {media_obs:.2f}%")
    print(f"   Desvio Padrão (Variação entre provas): {desvio_padrao:.2f}")
    print(f"   Diferença para o ideal: {media_obs - media_teorica:.2f}%")
    
    # 2. Execução
    stat, p_valor = ttest_1samp(amostra_a, popmean=media_teorica, alternative='less')
    
    print("\nPASSO 3: CÁLCULO ESTATÍSTICO (TESTE T)")
    print(f"   Estatística T: {stat:.4f}")
    print("   (Indica quantos desvios-padrão a sua média está abaixo do esperado)")
    
    print("\nPASSO 4: VEREDITO FINAL")
    print(f"   P-valor: {p_valor:.8f}")
    
    if p_valor < 0.05:
        print("   [!] REJEITA-SE H0.")
        print("   Conclusão: Existe evidência estatística robusta de que a banca EVITA a letra A.")
        print("   Dica Prática: Em chute cego, evite a letra A.")
    else:
        print("   [OK] ACEITA-SE H0.")
        print("   Conclusão: A letra A aparece dentro da normalidade.")

def plotar_boxplots_comparativos(df_dist):
    """
    Gera Boxplots para comparar a distribuição de todas as letras lado a lado.
    Permite ver claramente qual letra tem a mediana maior ou menor.
    """
    print_header("GERANDO GRÁFICOS DE QUARTIS (BOXPLOTS)")

    # 1. Boxplot para Certo/Errado
    df_ce = df_dist[df_dist['tipo_prova'] == 'CERTO_ERRADO']
    if not df_ce.empty:
        # Transforma de colunas (C, E) para linhas (Letra, Valor)
        melted_ce = df_ce.melt(id_vars=['concurso', 'cargo'], value_vars=['C', 'E'], 
                               var_name='Gabarito', value_name='Porcentagem')
        melted_ce = melted_ce.dropna()

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Gabarito', y='Porcentagem', data=melted_ce, palette=['green', '#d62728'])
        
        # Linha de referência 50%
        plt.axhline(50, color='blue', linestyle='--', linewidth=2, label='Equilíbrio (50%)')
        plt.title('Boxplot: Distribuição de Itens Certo vs Errado', fontsize=14)
        plt.ylabel('Porcentagem na Prova')
        plt.legend()
        plt.show()

    # 2. Boxplot para Múltipla Escolha (5 Alternativas)
    df_me5 = df_dist[(df_dist['tipo_prova'] == 'MULTIPLA_ESCOLHA') & (df_dist['qtd_alternativas'] == 5)]
    if not df_me5.empty:
        cols = [l for l in ['A','B','C','D','E'] if l in df_me5.columns]
        melted_me5 = df_me5.melt(id_vars=['concurso'], value_vars=cols, 
                                 var_name='Letra', value_name='Porcentagem')
        melted_me5 = melted_me5.dropna()

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Letra', y='Porcentagem', data=melted_me5, palette="husl")
        
        # Linha de referência 20%
        plt.axhline(20, color='blue', linestyle='--', linewidth=2, label='Teórico (20%)')
        plt.title('Boxplot: Comparativo de Alternativas (Provas de 5 Itens)', fontsize=14)
        plt.ylabel('Porcentagem na Prova')
        plt.legend()
        plt.show()

    # 3. Boxplot para Múltipla Escolha (4 Alternativas)
    df_me4 = df_dist[(df_dist['tipo_prova'] == 'MULTIPLA_ESCOLHA') & (df_dist['qtd_alternativas'] == 4)]
    if not df_me4.empty:
        cols = [l for l in ['A','B','C','D'] if l in df_me4.columns]
        melted_me4 = df_me4.melt(id_vars=['concurso'], value_vars=cols, 
                                 var_name='Letra', value_name='Porcentagem')
        melted_me4 = melted_me4.dropna()

        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Letra', y='Porcentagem', data=melted_me4, palette="husl")
        
        plt.axhline(25, color='blue', linestyle='--', linewidth=2, label='Teórico (25%)')
        plt.title('Boxplot: Comparativo de Alternativas (Provas de 4 Itens)', fontsize=14)
        plt.ylabel('Porcentagem na Prova')
        plt.legend()
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
        

        #Testes de hipótese
        relatorio_z_test_detalhado(df_classificado)
        relatorio_chi2_detalhado(df_classificado)
        relatorio_t_test_detalhado(df_classificado)

        if not df_ce.empty: 
            plotar_certo_errado(df_ce, df_counts_ce)
        if not df_me_5.empty: 
            plotar_multipla_escolha(df_me_5, df_counts_me_5, 5)
        if not df_me_4.empty: 
            plotar_multipla_escolha(df_me_4, df_counts_me_4, 4)

        # 2. NOVO: Gráficos de Boxplot (Quartis)
        plotar_boxplots_comparativos(df_dist)
    else:
        print("Nenhum dado encontrado.")