import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                              GradientBoostingClassifier, HistGradientBoostingClassifier, 
                              BaggingClassifier)
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import Counter


# Configuração de tela 
def config__tela():
    st.set_page_config(
        page_title='Classificação de Produtos', 
        page_icon='🏷️'
    )
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Classificação de Produtos 🏷️</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #FFD700;'>Tecnologia e Vestuário</h4>", unsafe_allow_html=True)
config__tela()

st.divider()

# Ocultar menus
def ocultar_menu():
    hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
    st.markdown(hide_st_style, unsafe_allow_html=True)
ocultar_menu()

# Carregando os dados
@st.cache_data
def carregamento_treino():
    data_url = "https://raw.githubusercontent.com/lsouzadasilva/datasets/main/treinamento_teste_produto.xlsx"
    df_treino = pd.read_excel(data_url, sheet_name=1)
    df_treino = df_treino.drop(columns='codigo')
    df_treino = df_treino.dropna()
    return df_treino
df_treino = carregamento_treino()

@st.cache_data
def carregamento_teste():
    data_url = "https://raw.githubusercontent.com/lsouzadasilva/datasets/main/treinamento_teste_produto.xlsx"
    df_teste = pd.read_excel(data_url, sheet_name=0, usecols=['marca', 'tipo'])
    df_teste = df_teste.dropna()
    return df_teste
df_teste = carregamento_teste()

def pre_processamento():
    codificador_marca = LabelEncoder()
    df_treino['marca'] = codificador_marca.fit_transform(df_treino['marca'])
    codificador_tipo = LabelEncoder()
    df_treino['tipo'] = codificador_tipo.fit_transform(df_treino['tipo'])
    return df_treino
df_treino = pre_processamento()

def conj_treinamento():
    y = df_treino['categoria']
    x = df_treino.drop(columns='categoria')
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)
    return x_treino, x_teste, y_treino, y_teste
x_treino, x_teste, y_treino, y_teste = conj_treinamento()

def modelo_ia(x_treino, y_treino):
    modelos = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "ExtraTrees": ExtraTreesClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "HistGradientBoosting": HistGradientBoostingClassifier(),
        "Bagging": BaggingClassifier(),
        "KNeighbors": KNeighborsClassifier(),
        "NearestCentroid": NearestCentroid(),
        "MLP": MLPClassifier()
    }
    st.sidebar.markdown("<h2 style='color: #A67C52;'>Seleção de Modelo</h2>", unsafe_allow_html=True)
    filtro_ia = st.sidebar.selectbox('', list(modelos.keys()))

    modelo_selecionado = modelos[filtro_ia]
    modelo_selecionado.fit(x_treino, y_treino)

    return modelo_selecionado
modelo_selecionado = modelo_ia(x_treino, y_treino)

def acuracidade():
    previsao = modelo_selecionado.predict(x_teste)
    acuracia = accuracy_score(y_teste, previsao)
    return acuracia
acuracia = acuracidade()

def card():
    nome_modelo = type(modelo_selecionado).__name__ 
    st.metric(
        label=f'Acuracidade: **{nome_modelo}**',
        value=f"{acuracia:.2f}%"
    )
card()

col1, col2 = st.columns(2)

with col1:
    def table_treino():
        st.markdown('## Treinamento')
        data_url = "https://raw.githubusercontent.com/lsouzadasilva/datasets/main/treinamento_teste_produto.xlsx"
        df_treino_table = pd.read_excel(data_url, sheet_name=1)
        df_treino_table = df_treino_table.drop(columns='codigo')
        df_treino_table = df_treino_table.dropna()
        table = df_treino_table
        table = st.dataframe(table, use_container_width=True, hide_index=True)
        return table
    table = table_treino()

with col2:
    def table_teste():
        st.markdown('## Teste')
        data_url = "https://raw.githubusercontent.com/lsouzadasilva/datasets/main/treinamento_teste_produto.xlsx"
        df_treino_table = pd.read_excel(data_url, sheet_name=0, usecols=['marca', 'tipo'])
        df_treino_table = df_treino_table.dropna()
        table = df_treino_table
        table = st.dataframe(table, use_container_width=True, hide_index=True)
        return table
    table = table_teste()

def resultado():
    codificador_marca_teste = LabelEncoder()
    df_teste['marca'] = codificador_marca_teste.fit_transform(df_teste['marca'])
    codificador_tipo_teste = LabelEncoder()
    df_teste['tipo'] = codificador_tipo_teste.fit_transform(df_teste['tipo'])
    resultados = modelo_selecionado.predict(df_teste)
    return resultados
resultados = resultado()

st.divider()
st.markdown(f'### Resultados')

def exibir_resultados():
    contagem = Counter(resultados)
    categorias = []
    valores = []
    
    for categoria, freq in contagem.items():
        categorias.append(categoria)
        valores.append(freq)
        st.progress(freq / sum(contagem.values()))
        st.write(f"**{categoria}:** {freq} produtos")
        
    if st.sidebar.button('Atualizar Dados 🔄'):
        st.rerun()
exibir_resultados()

def explicacao():
    st.sidebar.markdown("""
        **Bem-vindo à aplicação de Classificação de Produtos!** 🎯

        Esta aplicação realiza um estudo com a base de dados de **treinamento**, utilizando técnicas de **aprendizado de máquina** (Machine Learning) para treinar um modelo de classificação. O modelo é alimentado com dados de **marca** e **tipo** dos produtos e, com base nisso, classifica os produtos em diferentes **categorias**.

        A partir da base de dados de **teste**, o modelo faz previsões sobre quais categorias os produtos pertencem. Os resultados da previsão são apresentados logo abaixo, com a distribuição das categorias previstas.

        O modelo de classificação pode ser ajustado para diferentes algoritmos de aprendizado de máquina, permitindo testar e comparar a eficácia de cada um.
    """, unsafe_allow_html=True)

    st.sidebar.divider()
    st.sidebar.markdown("""
    **Desenvolvido por Leandro Souza**  
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in//leandro-souza-dados/)
    [![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/lsouzadasilva/classificacao_produtos_machine_learning)
""")
explicacao()
