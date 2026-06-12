import streamlit as st
from core import MLP, FUNCOES_ATIVACAO

# ── Configuração da Página ────────────────────────────────────────────────────
st.set_page_config(page_title="Simulador MLP", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
        color: #37352f;
        background-color: #ffffff;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: opacity 0.2s ease;
    }
    .stButton>button:hover { opacity: 0.9; color: white; }
    hr { border-top: 1px solid #ededed; margin: 1.5rem 0; }
    h1, h2, h3 { font-weight: 600; }
    div[data-testid="stMetricValue"] { font-family: monospace; color: #764ba2; }
    </style>
""", unsafe_allow_html=True)

# ── Dados ─────────────────────────────────────────────────────────────────────
X_dados = [[0, 0], [0, 1], [1, 0], [1, 1]]

funcoes_logicas = {
    'XOR':  [-1.0,  1.0,  1.0, -1.0],
    'AND':  [-1.0, -1.0, -1.0,  1.0],
    'OR':   [-1.0,  1.0,  1.0,  1.0],
    'NAND': [ 1.0,  1.0,  1.0, -1.0],
    'NOR':  [ 1.0, -1.0, -1.0, -1.0],
}

# ── Layout ────────────────────────────────────────────────────────────────────
st.title("Simulador de Retropropagação")
st.markdown("Implementação de funções lógicas via MLP otimizada com NumPy.")
st.markdown("---")

# ── Menu Lateral ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configurações do Modelo")
    funcao_selecionada = st.selectbox("Função Lógica", options=list(funcoes_logicas.keys()))
    nome_ativacao      = st.selectbox("Função de Ativação", options=list(FUNCOES_ATIVACAO.keys()))

    st.markdown("---")
    lr_input        = st.slider("Taxa de Aprendizagem", 0.05, 1.0, 0.2, 0.05)
    tol_input       = st.number_input("Erro Tolerado", value=0.001, format="%.4f")
    neuronios_input = st.slider("Neurônios (Camada Oculta)", 2, 10, 4, 1)
    seed_input      = st.number_input("Semente de Inicialização", value=42, step=1)

    st.markdown("---")
    btn_treinar = st.button("Executar Treinamento", use_container_width=True)

# ── Resultados ────────────────────────────────────────────────────────────────
if btn_treinar:
    y_alvo = funcoes_logicas[funcao_selecionada]

    rede = MLP(
        n_entrada=2,
        n_oculta=int(neuronios_input),
        taxa_aprendizagem=lr_input,
        seed=int(seed_input),
        nome_funcao=nome_ativacao,
    )

    with st.spinner("Calculando épocas..."):
        epocas, historico_erros = rede.treinar(
            X_dados, y_alvo, max_epocas=10000, tolerancia=tol_input
        )
        previsoes = rede.prever(X_dados)

    convergiu   = historico_erros[-1] < tol_input
    status_icon = "✅" if convergiu else "⚠️"

    col1, col2 = st.columns([1, 2.5])

    with col1:
        st.markdown("### Métricas")
        st.metric("Ciclos de Treino", epocas)
        st.metric("Erro Final", f"{historico_erros[-1]:.6f}")
        st.metric("Status", f"{status_icon} {'Convergiu' if convergiu else 'Não convergiu'}")

        st.markdown("### Validação")
        tabela = "| Entrada | Esperado | Calculado | Classe |\n| :--- | :--- | :--- | :--- |\n"
        for i in range(len(X_dados)):
            entrada_str = f"[{X_dados[i][0]}, {X_dados[i][1]}]"
            esperado    = 1 if y_alvo[i] > 0 else 0
            calculado   = f"{previsoes[i]:.4f}"
            classe      = 1 if previsoes[i] > 0 else 0
            ok          = "✅" if classe == esperado else "❌"
            tabela += f"| {entrada_str} | {esperado} | {calculado} | **{classe}** {ok} |\n"
        st.markdown(tabela)

    with col2:
        st.markdown("### Convergência do Erro")
        st.line_chart(historico_erros, height=400, color="#764ba2")

else:
    st.info("Configure os parâmetros no menu lateral e clique em **Executar Treinamento**.")