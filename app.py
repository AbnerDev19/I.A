import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# CONFIGURAÇÃO DE PÁGINA (Estilo Minimalista)
# ==========================================
st.set_page_config(page_title="Simulador MLP", layout="wide", initial_sidebar_state="expanded")

# CSS customizado para deixar a interface mais limpa
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 0rem;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# NÚCLEO MATEMÁTICO (Sua classe MLP)
# ==========================================
def sigmoide_bipolar(x):
    return 2 / (1 + np.exp(-x)) - 1

def d_sigmoide_bipolar(fx):
    return 0.5 * (1 - fx ** 2)

class MLP:
    def __init__(self, n_entrada=2, n_oculta=4, n_saida=1, taxa_aprendizagem=0.2, tolerancia=0.001, max_epocas=20000, seed=None):
        self.n_entrada = n_entrada
        self.n_oculta = n_oculta
        self.n_saida = n_saida
        self.lr = taxa_aprendizagem
        self.tol = tolerancia
        self.max_epocas = max_epocas
        if seed is not None:
            np.random.seed(seed)
        self.W1 = np.random.uniform(-1, 1, (self.n_entrada + 1, self.n_oculta))
        self.W2 = np.random.uniform(-1, 1, (self.n_oculta + 1, self.n_saida))
        self.historico_erro = []

    def _forward(self, x):
        x_bias = np.concatenate([[1.0], x])
        net_oculta = x_bias @ self.W1
        saida_oculta = sigmoide_bipolar(net_oculta)
        h_bias = np.concatenate([[1.0], saida_oculta])
        net_saida = h_bias @ self.W2
        saida_final = sigmoide_bipolar(net_saida)
        return x_bias, saida_oculta, h_bias, saida_final

    def treinar(self, X, y):
        for epoca in range(self.max_epocas):
            erro_total = 0.0
            for xi, yi in zip(X, y):
                x_bias, saida_oculta, h_bias, saida_final = self._forward(xi)
                erro = yi - saida_final
                delta_saida = erro * d_sigmoide_bipolar(saida_final)
                delta_oculta = d_sigmoide_bipolar(saida_oculta) * (self.W2[1:] @ delta_saida)
                self.W2 += self.lr * np.outer(h_bias, delta_saida)
                self.W1 += self.lr * np.outer(x_bias, delta_oculta)
                erro_total += 0.5 * np.sum(erro ** 2)
            self.historico_erro.append(erro_total)
            if erro_total < self.tol:
                return epoca + 1
        return self.max_epocas

    def prever(self, X):
        resultados = []
        for xi in X:
            _, _, _, saida = self._forward(xi)
            resultados.append(saida[0])
        return np.array(resultados)

# ==========================================
# DADOS DAS FUNÇÕES LÓGICAS
# ==========================================
X_dados = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
funcoes_logicas = {
    'XOR':  np.array([[-1.0], [1.0], [1.0], [-1.0]]),
    'AND':  np.array([[-1.0], [-1.0], [-1.0], [1.0]]),
    'OR':   np.array([[-1.0], [1.0], [1.0], [1.0]]),
    'NAND': np.array([[1.0], [1.0], [1.0], [-1.0]]),
    'NOR':  np.array([[1.0], [-1.0], [-1.0], [-1.0]]),
}

# ==========================================
# INTERFACE PRINCIPAL
# ==========================================
st.title("Simulador de Retropropagação")
st.markdown("---")

# Controles na barra lateral
with st.sidebar:
    st.subheader("Parâmetros da Rede")
    funcao_selecionada = st.selectbox("Função Lógica", options=list(funcoes_logicas.keys()))
    
    st.markdown("---")
    lr_input = st.slider("Taxa de Aprendizagem", min_value=0.1, max_value=1.0, value=0.2, step=0.1)
    tol_input = st.number_input("Erro Tolerado", value=0.001, format="%.4f")
    neuronios_input = st.slider("Neurônios na Camada Oculta", min_value=2, max_value=10, value=4, step=1)
    seed_input = st.number_input("Semente (Seed)", value=3, step=1)
    
    st.markdown("---")
    btn_treinar = st.button("Executar Treinamento", type="primary", use_container_width=True)

# Área de Resultados
if btn_treinar:
    y_alvo = funcoes_logicas[funcao_selecionada]
    
    # Instancia e treina
    rede = MLP(n_oculta=int(neuronios_input), taxa_aprendizagem=lr_input, tolerancia=tol_input, seed=int(seed_input))
    epocas = rede.treinar(X_dados, y_alvo)
    erro_final = rede.historico_erro[-1]
    
    # Previsões para a tabela
    previsoes = rede.prever(X_dados)
    
    # Divide a tela em duas colunas principais
    col_metricas, col_grafico = st.columns([1, 2])
    
    with col_metricas:
        st.metric(label="Ciclos até Convergência", value=f"{epocas} épocas")
        st.metric(label="Erro Quadrático Final", value=f"{erro_final:.6f}")
        
        st.markdown("### Validação")
        # Montar tabela de dados limpa usando Pandas
        df_resultados = pd.DataFrame({
            "Entrada A": X_dados[:, 0],
            "Entrada B": X_dados[:, 1],
            "Esperado": [1 if y > 0 else 0 for y in y_alvo.flatten()],
            "Saída da Rede": np.round(previsoes, 4),
            "Classificação": [1 if p > 0 else 0 for p in previsoes]
        })
        st.dataframe(df_resultados, hide_index=True, use_container_width=True)

    with col_grafico:
        # Gráfico minimalista
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Cores baseadas no estilo Gradient/Notion (simples e elegantes)
        ax.plot(rede.historico_erro, color='#6366f1', linewidth=2.5)
        
        # Removendo bordas desnecessárias (Spines)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#e5e7eb')
        ax.spines['bottom'].set_color('#e5e7eb')
        
        ax.set_title(f"Convergência do Erro - {funcao_selecionada}", fontsize=14, pad=15, loc='left', color='#374151')
        ax.set_xlabel("Épocas", color='#6b7280')
        ax.set_ylabel("Erro Total", color='#6b7280')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5, color='#d1d5db')
        
        st.pyplot(fig)

else:
    # Estado inicial vazio
    st.info("Ajuste os parâmetros na barra lateral e clique em **Executar Treinamento** para visualizar os resultados.")