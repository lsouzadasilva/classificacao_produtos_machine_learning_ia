# 🏷️ Classificação de Produtos - Machine Learning

Este projeto é uma aplicação interativa desenvolvida em **Python** utilizando **Streamlit**, que realiza a classificação de produtos com base em dados de marca e tipo, utilizando diversos algoritmos de **Machine Learning**.

## 🚀 Funcionalidades

- Carregamento automático de bases de **treinamento** e **teste** direto do GitHub
- Pré-processamento de dados com codificação de variáveis categóricas
- Treinamento de diversos modelos de **Classificação**:
  - Logistic Regression
  - Random Forest
  - Extra Trees
  - Gradient Boosting
  - HistGradient Boosting
  - Bagging Classifier
  - K-Neighbors
  - Nearest Centroid
  - Multi-Layer Perceptron
- Visualização da acurácia do modelo selecionado
- Exibição dos dados de treino e teste
- Distribuição gráfica das categorias previstas
- Atualização dinâmica dos dados

---

## 🎯 Como usar

### 1. Clone o repositório

bash
git clone https://github.com/seu-usuario/classificacao-produtos.git
cd classificacao-produtos

---

# 2. Instale as dependências
De preferência, use um ambiente virtual:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
Depois instale os pacotes:

bash
Copy
Edit
pip install -r requirements.txt

# 3. Execute o projeto
bash
Copy
Edit
streamlit run app.py

# 📄 Como funciona
A aplicação utiliza um conjunto de dados contendo informações de produtos do setor de Tecnologia e Vestuário.
O fluxo principal é:

Os dados de treinamento e teste são carregados automaticamente de um arquivo Excel disponível no GitHub.

Os dados são pré-processados com codificação de variáveis categóricas.

O usuário escolhe um modelo de classificação na barra lateral.

O modelo é treinado e avaliado, exibindo sua acuracidade.

Os dados de teste são classificados e os resultados são apresentados na tela.

# 📊 Exemplo de Resultado
Modelo selecionado: RandomForestClassifier
Acuracidade: 97.00%
Distribuição dos produtos classificados:
Categoria 1: 45 produtos
Categoria 2: 30 produtos
Categoria 3: 25 produtos

# 🧑‍💻 Tecnologias
Python

Streamlit

Pandas

Scikit-learn

# 👨‍🏫 Desenvolvido por
Leandro Souza

# ⭐️ Licença
Uso livre para fins acadêmicos, educacionais ou pessoais.
Contribuições são bem-vindas!
