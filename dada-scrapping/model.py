import sqlite3
from dataclasses import dataclass, asdict
from typing import List

@dataclass
class QuestaoGabarito:
    concurso: str
    numero_questao: int
    alternativa_correta: str
    materia: str = "Geral" # 'Geral' ou 'Específico'

class BancoDeDados:
    def __init__(self, nome_banco="concursos_data.db"):
        self.nome_banco = nome_banco
        self.dados_temporarios: List[QuestaoGabarito] = []
        self._inicializar_tabelas()

    def adicionar_questao(self, questao: QuestaoGabarito):
        # Guarda na memória RAM temporariamente até mandarmos salvar
        self.dados_temporarios.append(questao)

    def _inicializar_tabelas(self):
        """Cria a estrutura do banco se não existir"""
        conn = sqlite3.connect(self.nome_banco)
        cursor = conn.cursor()
        
        # Tabela 1: Concursos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS concursos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT UNIQUE
            )
        ''')

        # Tabela 2: Cargos (Ligada ao Concurso)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cargos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                concurso_id INTEGER,
                nome_cargo TEXT,
                tipo_prova TEXT, -- 'CERTO_ERRADO' ou 'MULTIPLA_ESCOLHA'
                FOREIGN KEY(concurso_id) REFERENCES concursos(id),
                UNIQUE(concurso_id, nome_cargo)
            )
        ''')

        # Tabela 3: Gabarito (Ligada ao Cargo)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS gabaritos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cargo_id INTEGER,
                numero_questao INTEGER,
                resposta TEXT,
                materia TEXT,
                FOREIGN KEY(cargo_id) REFERENCES cargos(id),
                UNIQUE(cargo_id, numero_questao) 
            )
        ''')
        
        conn.commit()
        conn.close()

    def _detectar_tipo_prova(self):
        """Analisa as respostas para descobrir se é Certo/Errado ou Múltipla Escolha"""
        respostas = set(q.alternativa_correta for q in self.dados_temporarios)
        
        # Se tiver letras como B, D ou A (e não for só C/E), é Múltipla Escolha
        # Nota: Cebraspe usa C (Certo) e E (Errado). Às vezes usa A/B para V/F, mas raro.
        # Se aparecer 'D' ou 'B' junto com outras, garantimos que é múltipla.
        letras_multipla = {'B', 'D'} 
        
        # Interseção: se tiver B ou D nas respostas, é múltipla escolha.
        if respostas.intersection(letras_multipla):
            return "MULTIPLA_ESCOLHA"
        
        # Se só tiver C, E, X (Anulada) ou A (às vezes a primeira letra), assume C/E
        return "CERTO_ERRADO"

    def salvar_no_banco(self, nome_concurso_raw, id_cargo_raw):
        """
        Pega os dados da memória RAM e persiste no SQLite.
        nome_concurso_raw: "AEB_24"
        id_cargo_raw: "1" (Será salvo como "Cargo 1")
        """
        if not self.dados_temporarios:
            return False

        conn = sqlite3.connect(self.nome_banco)
        cursor = conn.cursor()

        try:
            # 1. Inserir ou Pegar ID do Concurso
            cursor.execute("INSERT OR IGNORE INTO concursos (nome) VALUES (?)", (nome_concurso_raw,))
            cursor.execute("SELECT id FROM concursos WHERE nome = ?", (nome_concurso_raw,))
            concurso_id = cursor.fetchone()[0]

            # 2. Detectar Tipo de Prova
            tipo_prova = self._detectar_tipo_prova()
            nome_cargo = f"Cargo {id_cargo_raw}"

            # 3. Inserir ou Pegar ID do Cargo
            cursor.execute('''
                INSERT OR IGNORE INTO cargos (concurso_id, nome_cargo, tipo_prova) 
                VALUES (?, ?, ?)
            ''', (concurso_id, nome_cargo, tipo_prova))
            
            cursor.execute('''
                SELECT id FROM cargos 
                WHERE concurso_id = ? AND nome_cargo = ?
            ''', (concurso_id, nome_cargo))
            cargo_id = cursor.fetchone()[0]

            # 4. Inserir Questões (Bulk Insert para performance)
            lista_para_inserir = []
            for q in self.dados_temporarios:
                lista_para_inserir.append((
                    cargo_id, 
                    q.numero_questao, 
                    q.alternativa_correta, 
                    q.materia
                ))

            # 'INSERT OR REPLACE' atualiza se a questão já existir (útil se rodar 2 vezes)
            cursor.executemany('''
                INSERT OR REPLACE INTO gabaritos (cargo_id, numero_questao, resposta, materia)
                VALUES (?, ?, ?, ?)
            ''', lista_para_inserir)

            conn.commit()
            return True, len(lista_para_inserir), tipo_prova

        except Exception as e:
            print(f"Erro ao salvar no banco: {e}")
            conn.rollback()
            return False, 0, "ERRO"
        finally:
            conn.close()