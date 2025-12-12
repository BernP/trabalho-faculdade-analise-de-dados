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

    @staticmethod
    def simular_prova_unica(gabarito_real, opcoes, grupo, conhecimento, erro):
        total = len(gabarito_real)
        if total == 0: return 0, 0, []

        n_tentativas = int(total * conhecimento)
        indices_tentativa = np.random.choice(total, n_tentativas, replace=False)
        n_erros = int(n_tentativas * erro)
        indices_erros = np.random.choice(indices_tentativa, n_erros, replace=False)
        set_erros = set(indices_erros)
        set_tentativas = set(indices_tentativa)
        
        folha = [None] * total
        
        for i in range(total):
            if i in set_tentativas:
                if i in set_erros:
                    erradas = [op for op in opcoes if op != gabarito_real[i]]
                    folha[i] = random.choice(erradas)
                else:
                    folha[i] = gabarito_real[i]
        
        final, _ = EstrategiaChute.menos_marcada(folha, opcoes)
        
        acertos = sum(1 for c, r in zip(final, gabarito_real) if c == r)
        pct_acerto = acertos / total
        
        if grupo == "CERTO_ERRADO":
            erros_final = total - acertos
            pct_nota = (acertos - erros_final) / total
        else:
            pct_nota = pct_acerto
            
        return pct_acerto, pct_nota, folha

    # --- Legado para compatibilidade ---
    def _gerar_cenario(self, gabarito_real, opcoes, conhecimento, erro):
        _, _, folha = self.simular_prova_unica(gabarito_real, opcoes, "dummy", conhecimento, erro)
        return folha, 0, 0

    def _calcular_nota(self, cand, real, grupo):
        acertos = sum(1 for c, r in zip(cand, real) if c == r)
        return (acertos - (len(real)-acertos)) if grupo == "CERTO_ERRADO" else acertos

    def _calcular_aleatoria_esperada(self, total, acertos_ini, erros_ini, grupo, n_opcoes):
        return 0

    def gerar_dataset_completo(self, lista_conhecimento, lista_erro, n_simulacoes=1000):
        # (Mantido vazio para focar no laboratório, mas a classe existe para não quebrar imports)
        pass 

# ==========================================
# 3. LABORATÓRIO DE PROBABILIDADE (CORRIGIDO)
# ==========================================
class LaboratorioProbabilidade:
    def __init__(self, db_path):
        self.gerador = GeradorDeDados(db_path)
        print("\n[LAB] Carregando banco de provas...")
        self.cache_provas = []
        df = self.gerador._obter_todas_provas()
        for _, row in df.iterrows():
            gabarito = self.gerador._carregar_gabarito(row['id'])
            if not gabarito: continue
            opcoes, grupo = self.gerador._determinar_opcoes(row['tipo_prova'], gabarito)
            self.cache_provas.append({'gabarito': gabarito, 'opcoes': opcoes, 'grupo': grupo, 'nome': row['nome']})
        print(f"[LAB] Pronto. {len(self.cache_provas)} provas na memória.")

    def calcular_probabilidade_geometrica(self, conhecimento, erro, meta_acerto=0.92, n_sims_por_prova=600):
        """
        Calcula a probabilidade usando soma de logs para evitar underflow (virar zero).
        Se p=0, aplicamos 'suavização' (considera que 1 chance em N+1 é possível) ou ignoramos.
        Aqui vamos filtrar os ZEROS REAIS para dar a média das provas POSSÍVEIS.
        """
        probs_validas = []
        
        start_time = time.time()
        print(f"   > Simulando 600 tentativas para cada uma das {len(self.cache_provas)} provas...")
        
        for prova in self.cache_provas:
            sucessos = 0
            for _ in range(n_sims_por_prova):
                pct_acerto, _, _ = self.gerador.simular_prova_unica(
                    prova['gabarito'], prova['opcoes'], prova['grupo'], 
                    conhecimento, erro
                )
                if pct_acerto >= meta_acerto:
                    sucessos += 1
            
            p = sucessos / n_sims_por_prova
            
            # Só adicionamos na conta da média geométrica se p > 0
            if p > 0:
                probs_validas.append(p)
        
        # Média Geométrica via Logaritmos (Matematicamente equivalente a multiplicar e tirar raiz)
        # GM = exp( (sum(log(p))) / N )
        if not probs_validas:
            media_geo = 0.0
        else:
            soma_logs = sum(math.log(p) for p in probs_validas)
            media_geo = math.exp(soma_logs / len(probs_validas))
            
        tempo = time.time() - start_time
        
        # Estatísticas extras
        n_zeros = len(self.cache_provas) - len(probs_validas)
        pct_impossiveis = (n_zeros / len(self.cache_provas)) * 100
        
        return media_geo, pct_impossiveis, tempo

    def teste_1_comparacao_rigorosa(self):
        print("\n" + "#"*80)
        print(" TESTE 1: QUEM PASSA MAIS? (MÉDIA GEOMÉTRICA) ".center(80))
        print("#"*80)
        
        # Cenários
        perfil_a = {'c': 0.70, 'e': 0.10, 'n': 12, 'nome': "A (70% Saber / 12 Provas)"}
        perfil_b = {'c': 0.80, 'e': 0.05, 'n': 2, 'nome': "B (80% Saber / 2 Provas)"}
        meta = 0.92
        
        for p in [perfil_a, perfil_b]:
            print(f"--- Analisando {p['nome']} ---")
            p_geo, pct_imp, dt = self.calcular_probabilidade_geometrica(p['c'], p['e'], meta)
            
            # Chance acumulada: 1 - (1 - p_geo)^n
            chance_total = 1 - (1 - p_geo)**p['n']
            
            print(f"   > Provas 'Impossíveis' (0% chance): {pct_imp:.2f}% do banco")
            print(f"   > Probabilidade Média (Onde é possível): {p_geo:.6%}")
            print(f"   >>> RESULTADO FINAL ({p['n']} tentativas): {chance_total:.4%}")
            print(f"   Tempo: {dt:.2f}s\n")

    def teste_3_quantas_provas_rigoroso(self):
        print("\n" + "#"*80)
        print(" TESTE 3: PLANEJAMENTO (QUANTAS PROVAS?) ".center(80))
        print("#"*80)
        
        c, e, meta = 0.70, 0.10, 0.92
        print(f"Perfil: {c*100}% Conhecimento | {e*100}% Erro")
        
        p_base, pct_imp, _ = self.calcular_probabilidade_geometrica(c, e, meta)
        
        print(f"Probabilidade Base (Média Geométrica): {p_base:.6%}")
        
        if p_base < 0.000001:
            print(">>> CONCLUSÃO: A probabilidade é estatisticamente ZERO.")
            print("    O método de chute não cobre a lacuna de conhecimento para 92%.")
        else:
            # N = log(1 - target) / log(1 - p)
            n_nec = math.ceil(math.log(0.10) / math.log(1 - p_base))
            print(f">>> Necessário: {n_nec} PROVAS (para 90% de chance)")
            
            print("\nTabela de Confiança:")
            for n in [1, 10, 50, 100, n_nec]:
                chance = 1 - (1 - p_base)**n
                print(f"   - {n:3d} provas: {chance:6.2%}")

# ==========================================
# 4. EXECUÇÃO
# ==========================================
if __name__ == "__main__":
    DB_PATH = "../dada-scrapping/concursos_data.db"
    
    lab = LaboratorioProbabilidade(DB_PATH)
    lab.teste_1_comparacao_rigorosa()
    lab.teste_3_quantas_provas_rigoroso()