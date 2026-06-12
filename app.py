import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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

# ── Funções Auxiliares de Visualização (Parte 3 do Roteiro) ────────────────────
def gerar_grafico_ativacao(nome, funcao, cor_linha):
    x = np.linspace(-10, 10, 200)
    y = [funcao(val) for val in x]
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(x, y, color=cor_linha, linewidth=2.5, label=nome)
    
    ax.set_title(f"{nome}", fontsize=11, fontweight='bold', color='#37352f')
    ax.set_xlabel("Eixo X", fontsize=9, color='#666666')
    ax.set_ylabel("f(x)", fontsize=9, color='#666666')
    
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.axhline(0, color="#dbdbdb", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#dbdbdb", linewidth=0.8, linestyle="--")
    
    ax.legend(fontsize=8, loc="upper left")
    ax.tick_params(axis='both', labelsize=8)
    fig.tight_layout()
    
    return fig

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
st.markdown("Implementação de funções lógicas via MLP — arquitetura otimizada com NumPy.")
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
            X_dados, y_alvo, max_epocas=20000, tolerancia=tol_input
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

    # ── Experimento A ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Experimento A — Variando nº de neurônios na camada oculta")
    st.markdown("*(lr=0.2, tol=0.001, seed=42, XOR)*")

    tab_a = "| Neurônios | Épocas | Erro Final | Convergiu? |\n| :---: | :---: | :---: | :---: |\n"
    for n in [2, 3, 4, 5]:
        r = MLP(n_oculta=n, taxa_aprendizagem=0.2, seed=42)
        ep, hist = r.treinar(X_dados, funcoes_logicas['XOR'], tolerancia=0.001)
        conv = "✅" if hist[-1] < 0.001 else "❌"
        tab_a += f"| {n} | {ep} | {hist[-1]:.6f} | {conv} |\n"
    st.markdown(tab_a)

    # ── Experimento B ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Experimento B — Variando taxa de aprendizagem")
    st.markdown("*(n_oculta=4, tol=0.001, seed=42, XOR)*")

    tab_b = "| Taxa | Épocas | Erro Final | Convergiu? |\n| :---: | :---: | :---: | :---: |\n"
    for lr in [0.1, 0.2, 0.3, 0.4, 0.5]:
        r = MLP(n_oculta=4, taxa_aprendizagem=lr, seed=42)
        ep, hist = r.treinar(X_dados, funcoes_logicas['XOR'], tolerancia=0.001)
        conv = "✅" if hist[-1] < 0.001 else "❌"
        tab_b += f"| {lr} | {ep} | {hist[-1]:.6f} | {conv} |\n"
    st.markdown(tab_b)

    # ── Experimento C ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Experimento C — Influência da inicialização dos pesos (seeds)")
    st.markdown("*(n_oculta=4, lr=0.2, tol=0.001, XOR)*")

    tab_c = "| Seed | Épocas | Erro Final | Convergiu? |\n| :---: | :---: | :---: | :---: |\n"
    for s in [0, 1, 2, 3, 4]:
        r = MLP(n_oculta=4, taxa_aprendizagem=0.2, seed=s)
        ep, hist = r.treinar(X_dados, funcoes_logicas['XOR'], tolerancia=0.001)
        conv = "✅" if hist[-1] < 0.001 else "❌"
        tab_c += f"| {s} | {ep} | {hist[-1]:.6f} | {conv} |\n"
    st.markdown(tab_c)

else:
    st.info("Configure os parâmetros no menu lateral e clique em **Executar Treinamento**.")


# ── Nova Seção: Funções de Ativação (Parte 3 do Roteiro) ──────────────────────
st.markdown("---")
st.header("Funções de Ativação")
st.markdown("Visualização das curvas matemáticas utilizadas pelos neurônios da rede (Intervalo de -10 a 10).")

f_binaria, _ = FUNCOES_ATIVACAO['Sigmóide Binária']
f_bipolar, _ = FUNCOES_ATIVACAO['Sigmóide Bipolar']
f_tanh, _    = FUNCOES_ATIVACAO['Tangente Hiperbólica']

col_g1, col_g2, col_g3 = st.columns(3)

with col_g1:
    fig1 = gerar_grafico_ativacao("Sigmóide Binária", f_binaria, cor_linha="#667eea")
    st.pyplot(fig1, use_container_width=True) # <-- GRÁFICO FIXADO AQUI

with col_g2:
    fig2 = gerar_grafico_ativacao("Sigmóide Bipolar", f_bipolar, cor_linha="#764ba2")
    st.pyplot(fig2, use_container_width=True) # <-- GRÁFICO FIXADO AQUI

with col_g3:
    fig3 = gerar_grafico_ativacao("Tangente Hiperbólica", f_tanh, cor_linha="#ff7675")
    st.pyplot(fig3, use_container_width=True) # <-- GRÁFICO FIXADO AQUI