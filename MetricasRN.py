import os
import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D zooming
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import h5py
from tkinter import *
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.utils import custom_object_scope


# ler arquivo do modelo h5 de rede neural artificial
def ler_arquivo_h5(caminho_arquivo):
    modelo = tf.keras.models.load_model(caminho_arquivo)

    return modelo


# Função para ler o arquivo SWC e extrair os dados do neurônio
def ler_arquivo_swc(caminho_arquivo):
    dados_neuronio = []
    with open(caminho_arquivo, 'r') as f:
        for linha in f:
            if not linha.startswith('#'):
                dados_neuronio.append([float(i) for i in linha.strip().split()])
    return np.array(dados_neuronio)
    
   
#Funcao para  ler e extrair informações da rede de neuronios
def ler_arquivo_nt(caminho_arquivo):
    arquivo_rede = open(caminho_arquivo)
    linhas = arquivo_rede.readlines()
    dicionario = {}
    adjacencias = []
    i = 0
    while i < len(linhas):
        node_in_line = linhas[i]
        node_in_line = node_in_line.split()
        if node_in_line[1] != '0':
            salva = node_in_line[0]  # salva a chave da lista
            dicionario[node_in_line[0]] = 1  # cria a chave na lista (vertice no grafo)
            i = i + 1  # pula pra proxima linha
            node_in_line = linhas[i]
            node_in_line = node_in_line.split()
            dicionario[salva] = node_in_line  # adiciona a lista de adjcencias a vertice lida anteriormente
            i = i + 1  # pula pra proxima linha
        elif node_in_line[1] == '0':
            dicionario[node_in_line[0]] = []  # cria a a vertice que nao esta ligada a nenhuma outra vertice
            i = i + 1  # pula pra proxima linha

    # posicao dos nodoa no espaço  3d
    dict_pos3d = {}
    i = 0
    j = 0
    while i < len(linhas):
        node_in_line = linhas[i]
        node_in_line = node_in_line.split()

        if node_in_line[1] != '0':
            node_in_line = linhas[i]
            node_in_line = node_in_line.split()
            dict_pos3d[node_in_line[0]] = [float(node_in_line[2]), float(node_in_line[3]), float(node_in_line[4])]  # adiciona a lista de adjcencias a vertice lida anteriormente
            i = i + 2  # pula pra proxima linha
            j = j + 1
        elif node_in_line[1] == '0':
            dict_pos3d[node_in_line[0]] = [float(node_in_line[2]), float(node_in_line[3]), float(node_in_line[4])]
            # dicionario[node_in_line[0]] = [] #cria a a vertice que nao esta ligada a nenhuma outra vertic

            # transforma o dicionario de string para inteiro

            i = i + 1  # pula pra proxima linha

    grafo = nx.DiGraph(dicionario)

    return grafo, dict_pos3d


#Tela para selecionar se sera lida uma rede ou um neuronio individual
def tela_selecao_analise():
    def selecionar_tipo_analise():
        if escolha_analise.get() == "Neurônio Individual":
            selecionar_arquivo("swc")
        elif escolha_analise.get() == "Rede de Neurônios":
            selecionar_arquivo("nt")
        elif escolha_analise.get() == "Rede Neural Artificial":
            selecionar_arquivo("h5")  
            
    root = tk.Tk()
    root.title("Seleção de Análise")
    root.geometry("300x200")

    lbl_instrucao = ttk.Label(root, text="Selecione o tipo de análise:", font=("Helvetica", 12))
    lbl_instrucao.pack(pady=10)

    escolha_analise = tk.StringVar()
    rb_neuronio_individual = ttk.Radiobutton(root, text="Neurônio Individual", variable=escolha_analise, value="Neurônio Individual")
    rb_neuronio_individual.pack()

    rb_rede_neuronios = ttk.Radiobutton(root, text="Rede de Neurônios", variable=escolha_analise, value="Rede de Neurônios")
    rb_rede_neuronios.pack()

    rb_rede_neural = ttk.Radiobutton(root, text="Rede Neural Artificial", variable=escolha_analise, value="Rede Neural Artificial")
    rb_rede_neural.pack()

    btn_confirmar = ttk.Button(root, text="Confirmar", command=selecionar_tipo_analise)
    btn_confirmar.pack()

    root.mainloop()


def selecionar_arquivo(tipo_analise):
    arquivo_selecionado = filedialog.askopenfilename(filetypes=[(tipo_analise.upper() + " Files", "*." + tipo_analise)])
    if arquivo_selecionado:
        if tipo_analise == "swc":
            exibir_tela_informacoes_neuronio_individual(arquivo_selecionado)
        elif tipo_analise == "nt":
            grafo, posicoes = ler_arquivo_nt(arquivo_selecionado)  
            exibir_tela_informacoes_rede_neuronios(arquivo_selecionado, grafo, posicoes) 
        elif tipo_analise == "h5": 
            exibir_tela_medidas_rede_artificial(arquivo_selecionado)

def exibir_tela_selecao_arquivo():
    root = tk.Tk()
    root.title("Seleção de Arquivo")
    root.geometry("300x200")  # Ajuste a altura para acomodar a nova opção

    lbl_instrucao = ttk.Label(root, text="Selecione o tipo de arquivo para análise:", font=("Helvetica", 12))
    lbl_instrucao.pack(pady=10)

    btn_neuronio_individual = ttk.Button(root, text="Neurônio Individual (SWC)", command=lambda: selecionar_arquivo("swc"))
    btn_neuronio_individual.pack()

    btn_rede_neuronios = ttk.Button(root, text="Rede de Neurônios (NT)", command=lambda: selecionar_arquivo("nt"))
    btn_rede_neuronios.pack()

    btn_rede_neural = ttk.Button(root, text="Rede Neural Artificial (H5)", command=lambda: selecionar_arquivo("h5"))  # Adicione o botão para a rede neural artificial
    btn_rede_neural.pack()

    root.mainloop()



# Inicio das funcoes e metrificacao dos neronios em rede

def networkDensity(graph):
    return nx.density(graph)

def showNumberOfVertex(graph):
	return nx.number_of_nodes(graph)


def ClosenessCentrality(g):
    return nx.closeness_centrality(g)

def centDegreeIN(G):
    return nx.in_degree_centrality(G)

def centDegreeOUT(G):
    return nx.out_degree_centrality(G)

def betCentrality(g):
    return nx.betweenness_centrality(g, normalized=True, endpoints=False)

def averageClustering(G):
    return nx.average_clustering(G)

def stronglyConnected(g):
    return nx.number_strongly_connected_components(g)

def isStronglyConnected(g):
    return nx.is_strongly_connected(g)
    
       
# Inicio das funcoes de metrificacao dos neuronios individuais

def calcular_comprimento_neuronio(dados_neuronio):
    comprimento = 0
    for linha in dados_neuronio:
        if int(linha[6]) != -1:  # Se não for um ponto terminal
            pai = int(linha[6])
            comprimento += np.linalg.norm(dados_neuronio[int(pai)][2:5] - linha[2:5])
    return comprimento

# Função para contar o número de ramos
def contar_ramos(dados_neuronio):
    ramos = set()
    for linha in dados_neuronio:
        pai = int(linha[6])
        if pai != -1:  # Se não for um ponto terminal
            ramos.add(pai)
    return len(ramos)

# Função para contar o número de terminações
def contar_terminacoes(dados_neuronio):
    terminacoes = 0
    pontos = set(linha[0] for linha in dados_neuronio)
    for linha in dados_neuronio:
        ponto = linha[6]
        if ponto not in pontos:  # Se for um ponto terminal
            terminacoes += 1
    return terminacoes

# Função para contar o número de interseções
def contar_intersecoes(dados_neuronio):
    intersecoes = 0
    pais = set()
    for linha in dados_neuronio:
        pai = int(linha[6])
        if pai != -1:
            if pai in pais:
                intersecoes += 1
            else:
                pais.add(pai)
    return intersecoes

# Função para calcular a média dos graus dos nós
def calcular_media_grau_nos(dados_neuronio):
    G = nx.Graph()
    for linha in dados_neuronio:
        G.add_node(int(linha[0]))
        if int(linha[6]) != -1:
            G.add_edge(int(linha[0]), int(linha[6]))
    graus = [grau for nó, grau in G.degree()]
    return np.mean(graus) if graus else 0




# Função para extrair as posições do neurônio
def extrair_posicoes_neuronio(dados_neuronio):
    pos = {}
    for linha in dados_neuronio:
        pos[int(linha[0])] = (linha[2], linha[3], linha[4])
    return pos

################ Inicio da extracao de medidas da rede neural artificial########

def extrair_medidas_rede_neural(nodes, edges):
    # Criação do grafo
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Calcular medidas
    density = nx.density(G)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    medidas = {
        "Densidade da Rede": density,
        "Número de Nós": num_nodes,
        "Número de Arestas": num_edges
    }

    return medidas


def plotGraph3d(root, dados_neuronio, pos):
    G = nx.Graph()
    for linha in dados_neuronio:
        G.add_node(int(linha[0]), pos=(linha[2], linha[3], linha[4]))
    for linha in dados_neuronio:
        if int(linha[6]) != -1:
            G.add_edge(int(linha[0]), int(linha[6]))
    fig = plt.figure(figsize=(8, 6), facecolor="#040c24")
    ax: Axes3D = fig.add_subplot(111, projection="3d")  
    for u, v in G.edges():
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], [pos[u][2], pos[v][2]], color='#04eaa6')
    x = [pos[node][0] for node in G.nodes()]
    y = [pos[node][1] for node in G.nodes()]
    z = [pos[node][2] for node in G.nodes()]
    ax.scatter(x, y, z, color='#da8ee7', s=3, depthshade=True)
    ax.grid(color='none', linewidth=0.5, linestyle='dashed', alpha=0.5)
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])
    ax.set_xlabel("X", fontweight='bold', color='white')
    ax.set_ylabel("Y", fontweight='bold', color='white')
    ax.set_zlabel("Z", fontweight='bold', color='white')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_facecolor("#040c24")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)


def exibir_tela_informacoes_neuronio_individual(arquivo_selecionado):
    def visualizar_3d():
        dados_neuronio = ler_arquivo_swc(arquivo_selecionado)
        pos = extrair_posicoes_neuronio(dados_neuronio)
        plotGraph3d(root, dados_neuronio, pos)
    
    def selecionar_outro_arquivo():
        root.destroy()
        tela_selecao_analise()
    
    dados_neuronio = ler_arquivo_swc(arquivo_selecionado)
    comprimento = calcular_comprimento_neuronio(dados_neuronio)
    num_ramos = contar_ramos(dados_neuronio)
    num_terminacoes = contar_terminacoes(dados_neuronio)
    num_intersecoes = contar_intersecoes(dados_neuronio)
    media_grau_nos = calcular_media_grau_nos(dados_neuronio)

    root = tk.Tk()
    root.title("Informações do Neurônio")
    largura_tela = root.winfo_screenwidth()
    altura_tela = root.winfo_screenheight()
    root.geometry(f"{largura_tela}x{altura_tela}")
    fr = tk.Frame(root, bg="#040c24")
    fr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)


    lbl_info_neuronio = ttk.Label(fr, text="Informações do Neurônio:", font=("Helvetica", 33), foreground="#40e0d0", background="#040c24")
    lbl_info_neuronio.pack(pady=10, anchor="w")
    lbl_num_pontos = ttk.Label(fr, text=f"Número de Pontos: {len(dados_neuronio)}", foreground="white", background="#161e33", font=("Helvetica", 22), borderwidth=10)
    lbl_num_pontos.pack(anchor="w")
    lbl_comprimento = ttk.Label(fr, text=f"Comprimento do Neurônio: {comprimento:.2f} unidades", foreground="white", background="#161e33", font=("Helvetica", 22), borderwidth=10)
    lbl_comprimento.pack(anchor="w")
    lbl_num_ramos = ttk.Label(fr, text=f"Número de Ramos: {num_ramos}", foreground="white", background="#161e33", font=("Helvetica", 22), borderwidth=10)
    lbl_num_ramos.pack(anchor="w")
    lbl_num_terminacoes = ttk.Label(fr, text=f"Número de Terminações: {num_terminacoes}", foreground="white", background="#161e33", font=("Helvetica", 22), borderwidth=10)
    lbl_num_terminacoes.pack(anchor="w")
    lbl_num_intersecoes = ttk.Label(fr, text=f"Número de Interseções: {num_intersecoes}", foreground="white", background="#161e33", font=("Helvetica", 22), borderwidth=10)
    lbl_num_intersecoes.pack(anchor="w")
    lbl_media_grau_nos = ttk.Label(fr, text=f"Média de Grau dos Nós: {media_grau_nos:.2f}", foreground="white", background="#161e33", font=("Helvetica", 22), borderwidth=10)
    lbl_media_grau_nos.pack(anchor="w")
    pos = extrair_posicoes_neuronio(dados_neuronio)
    canvas = plotGraph3d(root, dados_neuronio, pos)
    canvas.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=canvas.yview)

    root.mainloop()


def plotar_rede_neuronios(root, G, pos):
    node_xyz = np.array([pos[v] for v in G])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])
    fig = plt.figure(facecolor="#040c24")
    ax = fig.add_subplot(111, projection="3d")
    graph = nx.Graph(G)
    graph_degree = nx.degree(graph)
    color = [("#d5396f" if node[1] == 0 else "#da8ee7") for node in graph_degree]
    ax.scatter(*node_xyz.T, s=50, ec="w", color=color)
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="#04eaa6")
    ax.grid(True)
    for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
        dim.set_ticks([])
    ax.set_xlabel("X", fontweight='bold', color='white')
    ax.set_ylabel("Y", fontweight='bold', color='white')
    ax.set_zlabel("Z", fontweight='bold', color='white')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_facecolor("#040c24")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)


def exibir_tela_informacoes_rede_neuronios(arquivo_selecionado, grafo, posicoes):
    root = tk.Tk()
    root.title("Informações da Rede de Neurônios")
    largura_tela = root.winfo_screenwidth()
    altura_tela = root.winfo_screenheight()
    root.geometry(f"{largura_tela}x{altura_tela}")
    
    # Criação do frame principal
    fr = tk.Frame(root, bg="#040c24")
    fr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    # Função para selecionar outro arquivo
    def selecionar_outro_arquivo():
        root.destroy()
        tela_selecao_analise()
    
    # Criação do Canvas
    canvas = tk.Canvas(fr, bg="#040c24")
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    

    # Frame para conter os elementos
    frame_content = tk.Frame(canvas, bg="#040c24")
    canvas.create_window((0, 0), window=frame_content, anchor='nw')
    
    # Função para configurar o tamanho do canvas
    def configure_canvas(event):
        canvas.configure(scrollregion=canvas.bbox("all"), width=event.width, height=event.height)
    frame_content.bind("<Configure>", configure_canvas)
    

    graph = nx.DiGraph(grafo)
    density = networkDensity(graph)
    closeness_centrality = ClosenessCentrality(graph)
    in_degree_centrality = centDegreeIN(graph)
    out_degree_centrality = centDegreeOUT(graph)
    betweenness_centrality = betCentrality(graph)
    avg_clustering = averageClustering(graph)
    num_strongly_connected = stronglyConnected(graph)
    is_strongly_connected = isStronglyConnected(graph)
    number_nodes = showNumberOfVertex(graph)
    
    ttk.Label(frame_content, text="Métricas da Rede de Neurônios:", font=("Helvetica", 33), foreground="#40e0d0", background="#040c24").pack(anchor="w", pady=(10, 5))

    # Configurar e criar rótulos para as métricas da rede de neurônios
    ttk.Label(frame_content, text=f"Densidade da Rede: {density:.4f}", foreground="white", font=("Helvetica", 22), background="#161e33", borderwidth=10).pack(anchor="w", pady=5)
    ttk.Label(frame_content, text=f"Número de Nós na Rede: {number_nodes}", foreground="white", font=("Helvetica", 22), background="#161e33", borderwidth=10).pack(anchor="w", pady=5)
    ttk.Label(frame_content, text=f"Média de Aglomeração: {avg_clustering:.4f}", foreground="white", font=("Helvetica", 22), background="#161e33", borderwidth=10).pack(anchor="w", pady=5)
    ttk.Label(frame_content, text=f"Número de Componentes Fortemente Conectados: {num_strongly_connected}", foreground="white", font=("Helvetica", 22), background="#161e33", borderwidth=10).pack(anchor="w", pady=5)
    ttk.Label(frame_content, text=f"A Rede é Fortemente Conectada? {'Sim' if is_strongly_connected else 'Não'}", font=("Helvetica", 22), foreground="white", background="#161e33", borderwidth=10).pack(anchor="w", pady=5)

    
    def abrir_janela_in_degree():
        janela = tk.Toplevel(root)
        janela.title("In-Degree Centralidade")
        fr_in_degree = tk.Frame(janela, bg="#040c24")
        fr_in_degree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(fr_in_degree, text="In-Degree Centralidade", font=("Helvetica", 33), foreground="white", background="#040c24").pack(anchor="w", pady=(20, 10))
        for node, centrality in in_degree_centrality.items():
            ttk.Label(fr_in_degree, text=f"Nó {node}: {centrality:.4f}", font=("Helvetica", 22), foreground="white", background="#040c24").pack(anchor="w")

    def abrir_janela_out_degree():
        janela = tk.Toplevel(root)
        janela.title("Out-Degree Centralidade")
        fr_out_degree = tk.Frame(janela, bg="#040c24")
        fr_out_degree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(fr_out_degree, text="Out-Degree Centralidade", font=("Helvetica", 33), foreground="white", background="#040c24").pack(anchor="w", pady=(20, 10))
        for node, centrality in out_degree_centrality.items():
            ttk.Label(fr_out_degree, text=f"Nó {node}: {centrality:.4f}", font=("Helvetica", 22), foreground="white", background="#040c24").pack(anchor="w")

    def abrir_janela_closeness():
        janela = tk.Toplevel(root)
        janela.title("Closeness Centralidade")
        fr_closeness = tk.Frame(janela, bg="#040c24")
        fr_closeness.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(fr_closeness, text="Closeness Centralidade", font=("Helvetica", 33), foreground="white", background="#040c24").pack(anchor="w", pady=(20, 10))
        for node, centrality in closeness_centrality.items():
            ttk.Label(fr_closeness, text=f"Nó {node}: {centrality:.4f}", font=("Helvetica", 22), foreground="white", background="#040c24").pack(anchor="w")

    def abrir_janela_betweenness():
        janela = tk.Toplevel(root)
        janela.title("Betweenness Centralidade")
        fr_betweenness = tk.Frame(janela, bg="#040c24")
        fr_betweenness.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(fr_betweenness, text="Betweenness Centralidade", font=("Helvetica", 33), foreground="white", background="#040c24").pack(anchor="w", pady=(20, 10))
        for node, centrality in betweenness_centrality.items():
            ttk.Label(fr_betweenness, text=f"Nó {node}: {centrality:.4f}", font=("Helvetica", 22), foreground="white", background="#040c24").pack(anchor="w")

    indegreeB = Button(root, text="In-Degree Centrality", border=0, height=2, width=30, font=("consolas", 20), command=abrir_janela_in_degree)
    indegreeB.place(x=10, y=490)
    indegreeB.config(bg="#161e33", fg="#FFFFFF")
    
    outdegreeB = Button(root, text="Out-Degree Centrality", border=0, height=2, width=30, font=("consolas", 20), command=abrir_janela_out_degree)
    outdegreeB.place(x=10, y=530)
    outdegreeB.config(bg="#161e33", fg="#FFFFFF")
    
    closenessB = Button(root, text="Closeness Centrality", border=0, height=2, width=30, font=("consolas", 20), command=abrir_janela_closeness)
    closenessB.place(x=10, y=570)
    closenessB.config(bg="#161e33", fg="#FFFFFF")
    
    betweennessB = Button(root, text="Betweenness Centrality", border=0, height=2, width=30, font=("consolas", 20), command=abrir_janela_betweenness)
    betweennessB.place(x=10, y=610)
    betweennessB.config(bg="#161e33", fg="#FFFFFF")


    plotar_rede_neuronios(root, graph, posicoes)


    root.mainloop()


def exibir_pesos_completos(modelo):
    distribuicao_pesos = [peso.numpy() for peso in modelo.trainable_weights]
    distribuicao_pesos_texto = "\n".join([f"Camada {i+1}: {peso}" for i, peso in enumerate(distribuicao_pesos)])
    
    # Criando uma nova janela para exibir os pesos
    root_pesos = tk.Tk()
    root_pesos.title("Pesos da Rede Neural")
    root_pesos.configure(bg="#040c24")

    text_area = tk.Text(root_pesos, height=20, width=80, bg="#040c24", fg="white", font=("Helvetica", 22))
    text_area.insert(tk.END, distribuicao_pesos_texto)
    text_area.pack()


def exibir_tela_medidas_rede_artificial(arquivo_selecionado):
    modelo = ler_arquivo_h5(arquivo_selecionado)
    numero_camadas = len(modelo.layers)
    G = nx.Graph()
    for camada in modelo.layers:
        nome_camada = camada.name
        G.add_node(nome_camada)
    for camada_origem, camada_destino in zip(modelo.layers[:-1], modelo.layers[1:]):
        nome_origem = camada_origem.name
        nome_destino = camada_destino.name
        G.add_edge(nome_origem, nome_destino)

    root = tk.Tk()
    root.title("Informações da Rede Artificial")
    largura_tela = root.winfo_screenwidth()
    altura_tela = root.winfo_screenheight()
    root.geometry(f"{largura_tela}x{altura_tela}")

    # Frame principal
    fr = tk.Frame(root, bg="#040c24")
    fr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Frame para conter as informações
    info_frame = tk.Frame(fr, bg="#040c24")
    info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Frame para conter o gráfico
    graph_frame = tk.Frame(fr, bg="#040c24")
    graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    ttk.Label(info_frame, text="Informações da Rede Artificial", font=("Helvetica", 33), foreground="#40e0d0", background="#040c24").pack(anchor="w", pady=(20, 10))
    ttk.Label(info_frame, text=f"Número de Camadas: {numero_camadas}", foreground="white", background="#161e33", font=("Helvetica", 22), borderwidth=10).pack(anchor="w", pady=5)
    def exibir_pesos():
        exibir_pesos_completos(modelo)

    
    for camada in modelo.layers:
        nome_camada = camada.name
        numero_neuronios = camada.units
        ttk.Label(info_frame, text=f"Camada {nome_camada}: {numero_neuronios} neurônios", foreground="white", background="#161e33", font=("Helvetica", 22), borderwidth=10).pack(anchor="w", pady=5)

    numero_parametros_treinaveis = np.sum([tf.size(w).numpy() for w in modelo.trainable_weights])
    taxa_aprendizado = modelo.optimizer.learning_rate.numpy() 
    distribuicao_pesos = [np.mean(w.numpy()) for w in modelo.trainable_weights]
    distribuicao_pesos_texto = "\n".join([f"{i+1}: {peso}" for i, peso in enumerate(distribuicao_pesos)])

    ttk.Label(info_frame, text=f"Número de Parâmetros Treináveis: {numero_parametros_treinaveis}", foreground="white", background="#161e33", font=("Helvetica", 22), borderwidth=10).pack(anchor="w", pady=5)
    ttk.Label(info_frame, text=f"Taxa de Aprendizado: {taxa_aprendizado}", foreground="white", background="#161e33", font=("Helvetica", 22), borderwidth=10).pack(anchor="w", pady=5)
   
    pesosB = Button(root, text="Exibir Pesos da rede", border=0, height=2, width=30, font=("consolas", 20), command=exibir_pesos)
    pesosB.place(x=10, y=450)
    pesosB.config(bg="#161e33", fg="#FFFFFF")

   
    def draw_network():
        fig, ax = plt.subplots(figsize=(7, 4), tight_layout=True, facecolor='#040c24')
        ax.set_facecolor('#040c24')
        ax.set_title('Neuronios da rede artificial', color='white')
        cores = ['#FFB6C1', '#DA70D6', '#ADD8E6']

        for i, camada in enumerate(modelo.layers):
            x_coords = np.ones(camada.units) * i
            if i == numero_camadas - 1:
                deslocamento_vertical = (1 - camada.units) / 2
                y_coords = np.linspace(deslocamento_vertical, 1 - deslocamento_vertical, camada.units)
            else:
                y_coords = np.linspace(0, 1, camada.units)

            ax.scatter(x_coords, y_coords, label=f'Layer {i+1}', s=100, facecolor=cores[i])  

            if i < numero_camadas - 1:
                next_layer = modelo.layers[i+1]
                for j in range(camada.units):
                    for k in range(next_layer.units):
                        ax.plot([i, i+1], [y_coords[j], y_coords[k]], color='white', alpha=0.5)

        ax.set_xlabel('Layers', color='white')
        ax.set_ylabel('Neurons', color='white')
        ax.set_xticks(range(numero_camadas))
        ax.set_xticklabels([f'Layer {i+1}' for i in range(numero_camadas)], color='white')
        ax.legend(facecolor='#040c24', edgecolor='white', labelcolor='white')
        ax.grid(True, color='white')

        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=0, pady=0, ipadx=0, ipady=0, anchor='ne')

  
        
    draw_network()
    
    root.mainloop()


tela_selecao_analise()

