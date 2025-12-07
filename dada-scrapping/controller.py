import io
import re
import time
import unicodedata
import os
import sqlite3

import requests
import pdfplumber
from bs4 import BeautifulSoup
from collections import defaultdict

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from model import QuestaoGabarito, BancoDeDados
from view import TerminalView

class CebraspeCrawler:
    def __init__(self):
        self.view = TerminalView()

    def _normalizar_texto(self, texto):
        return ''.join(c for c in unicodedata.normalize('NFD', texto) 
                      if unicodedata.category(c) != 'Mn').upper()

    def _iniciar_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--log-level=3")
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=chrome_options)

    def listar_todos_concursos(self, url_encerrados):
        self.view.mostrar_status(f"Acessando listagem: {url_encerrados}...")
        driver = self._iniciar_driver()
        try:
            driver.get(url_encerrados)
            time.sleep(6)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            links = soup.find_all('a')
            urls_concursos = set()

            for link in links:
                href = link.get('href', '')
                if '/concursos/' in href and not href.endswith('#'):
                    if not any(x in href for x in ['encerrado', 'vigente', 'proximos']):
                        if href.startswith('/'):
                            href = f"https://www.cebraspe.org.br{href}"
                        urls_concursos.add(href)
            
            lista = list(urls_concursos)
            print(f"✅ Lista carregada: {len(lista)} concursos encontrados.")
            return lista
        except Exception as e:
            self.view.mostrar_erro(f"Erro ao listar: {e}")
            return []
        finally:
            driver.quit()

    def mapear_cargos(self, url_pagina_concurso):
        self.view.mostrar_status(f"Mapeando: {url_pagina_concurso}...")
        driver = self._iniciar_driver()
        try:
            driver.get(url_pagina_concurso)
            time.sleep(4)
            html = driver.page_source
            driver.quit()

            soup = BeautifulSoup(html, 'html.parser')
            links = soup.find_all('a')
            cargos = defaultdict(dict)
            
            termos_validos = ["GABARITO", "DEFINITIVO"]
            
            for link in links:
                texto_original = link.get_text(" ", strip=True)
                texto_norm = self._normalizar_texto(texto_original)
                href = link.get('href', '')
                
                if not href or len(href) < 5 or not href.lower().endswith('.pdf'):
                    continue
                if href.startswith('/'): href = f"https://www.cebraspe.org.br{href}"

                if all(termo in texto_norm for termo in termos_validos):
                    tipo = "basico"
                    if "ESPECIFICO" in texto_norm: tipo = "especifico"
                    
                    match = re.search(r'CARGOS?.{0,30}?([\d\s,eE]+)', texto_norm)
                    lista_ids = []
                    if match:
                        trecho = match.group(1)
                        lista_ids = re.findall(r'\d+', trecho)
                    
                    if not lista_ids: lista_ids = re.findall(r'CARGO_?(\d+)', href.upper())
                    if not lista_ids: lista_ids = ['1'] 

                    if lista_ids:
                        for cid in lista_ids:
                            cargos[str(int(cid))][tipo] = href
            return cargos
        except:
            try: driver.quit()
            except: pass
            return {}

class PDFProcessor:
    def __init__(self):
        self.db = BancoDeDados()
        self.view = TerminalView()

    def limpar_memoria(self):
        self.db = BancoDeDados()

    def processar_pdf(self, url_pdf, nome_concurso, tipo_materia):
        try:
            self.view.mostrar_status(f"   -> Baixando {tipo_materia}...")
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url_pdf, headers=headers, verify=False, timeout=30)
            if response.status_code != 200: return

            arquivo_pdf = io.BytesIO(response.content)

            with pdfplumber.open(arquivo_pdf) as pdf:
                for i, pagina in enumerate(pdf.pages):
                    # ESTRATÉGIA 1: Tabelas (Se houver linhas desenhadas)
                    tabelas = pagina.extract_tables()
                    if tabelas:
                        for tabela in tabelas:
                            self._estrategia_tabela(tabela, nome_concurso, tipo_materia)
                    
                    # Se tabelas não funcionaram bem, tenta texto
                    texto = pagina.extract_text()
                    if texto:
                        # ESTRATÉGIA 2: Horizontal (Cebraspe Clássico)
                        # Onde uma linha tem "1 2 3" e a debaixo tem "C E C"
                        achou_horizontal = self._estrategia_horizontal(texto, nome_concurso, tipo_materia)
                        
                        # ESTRATÉGIA 3: Vertical/Regex (Se Horizontal falhar)
                        if not achou_horizontal:
                            self._estrategia_regex_vertical(texto, nome_concurso, tipo_materia)

        except Exception as e:
            print(f"      [Erro Leitura] {str(e)[:50]}")

    def _estrategia_tabela(self, tabela, concurso, materia):
        for linha in tabela:
            # Filtra None e vazios
            linha = [x for x in linha if x]
            if len(linha) < 2: continue
            
            # Formato: [Numero, Letra]
            if linha[0].isdigit() and len(linha[1]) == 1:
                self._add(concurso, linha[0], linha[1], materia)

    def _estrategia_horizontal(self, texto, concurso, materia):
        """
        Analisa o texto linha por linha procurando o padrão:
        Linha A: ... 1  2  3  4 ... (Muitos números)
        Linha B: ... C  E  C  X ... (Muitas letras logo abaixo)
        """
        linhas = texto.split('\n')
        questoes_adicionadas = 0
        
        for i in range(len(linhas) - 1):
            linha_atual = linhas[i].strip()
            linha_prox = linhas[i+1].strip()
            
            # Encontra todos os números na linha de cima
            # \b garante que não pegue pedaços de palavras
            numeros = re.findall(r'\b(\d{1,3})\b', linha_atual)
            
            # Encontra todas as letras na linha de baixo (A-E, C, E, X)
            letras = re.findall(r'\b([A-E]|[CX])\b', linha_prox)
            
            # REGRA: Para ser considerado um bloco horizontal válido, 
            # deve haver pelo menos 5 correspondências sequenciais
            if len(numeros) > 4 and len(letras) > 4:
                # O PDFPlumber as vezes desalinha, então pegamos o mínimo comum
                qtd = min(len(numeros), len(letras))
                
                # Validação Extra: As letras devem estar mais ou menos na mesma quantidade dos números
                # Se tiver 20 números e 2 letras, é falso positivo.
                if abs(len(numeros) - len(letras)) > 5:
                    continue

                for k in range(qtd):
                    self._add(concurso, numeros[k], letras[k], materia)
                    questoes_adicionadas += 1
        
        return questoes_adicionadas > 0

    def _estrategia_regex_vertical(self, texto, concurso, materia):
        """
        Procura o padrão '1 C' ou '1. C' no texto corrido
        """
        # Padrão: Numero + (ponto/traço opcional) + Espaço + Letra
        padrao = r"(?<!\d)(\d{1,3})\s*[\.\-]?\s+([A-E]|[CX])(?!\w)"
        resultados = re.findall(padrao, texto)
        
        for numero, letra in resultados:
            self._add(concurso, numero, letra, materia)

    def _add(self, concurso, num, letra, materia):
        try:
            n = int(num)
            # Filtro: Questões válidas (1 a 250) e ignora anos (2020, 2023)
            if 0 < n < 250:
                q = QuestaoGabarito(concurso, n, letra.upper(), materia)
                self.db.adicionar_questao(q)
        except:
            pass

    def salvar_final(self, nome_concurso, id_cargo):
        if self.db.dados_temporarios:
            sucesso, qtd, tipo = self.db.salvar_no_banco(nome_concurso, id_cargo)
            if sucesso:
                print(f"      💾 Salvo: Cargo {id_cargo} | {qtd} questões ({tipo})")
        else:
            print(f"      ⚠️  Nada extraído para Cargo {id_cargo}")