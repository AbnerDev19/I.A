import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from core import MLP

# =====================================================================
# Experimentos para o Artigo Científico - Função XOR
# =====================================================================

# Dados de entrada XOR
X_dados = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_alvo = np.array([-1.0, 1.0, 1.0, -1.0]) # Bipolar para convergir melhor

print("Iniciando Experimentos para o Artigo...\n")

# --- Experimento A: Influência do Número de Neurônios na Camada Oculta ---
neuronios_teste = [2, 3, 4, 5, 8]
resultados_n = []

plt.figure(figsize=(10, 6))
for n in neuronios_teste:
    # Cria e treina a rede importada do core.py
    mlp = MLP(n_entrada=2, n_oculta=n, taxa_aprendizagem=0.2, seed=42)
    epocas, erros = mlp.treinar(X_dados, Y_alvo, max_epocas=5000, tolerancia=0.001)
    
    # Plota a linha deste teste no gráfico
    plt.plot(erros, label=f'{n} Neurônios')
    resultados_n.append({'Neurônios': n, 'Épocas até Convergência': epocas, 'Erro Final': round(erros[-1], 6)})

# Estilização do Gráfico
plt.title("Convergência do Erro variando Neurônios na Camada Oculta")
plt.xlabel("Épocas")
plt.ylabel("Erro Quadrático Médio")
plt.legend()
plt.grid(True)
plt.savefig("grafico_neuronios.png")
print("✅ Gráfico 'grafico_neuronios.png' salvo com sucesso!")

# Tabela com Pandas
df_n = pd.DataFrame(resultados_n)
print("\nTabela 1: Influência do Número de Neurônios (XOR)")
print(df_n.to_markdown(index=False))
print("-" * 50)


# --- Experimento B: Influência da Taxa de Aprendizagem (Learning Rate) ---
taxas_teste = [0.05, 0.1, 0.3, 0.6, 0.9]
resultados_lr = []

plt.figure(figsize=(10, 6))
for lr in taxas_teste:
    mlp = MLP(n_entrada=2, n_oculta=4, taxa_aprendizagem=lr, seed=42)
    epocas, erros = mlp.treinar(X_dados, Y_alvo, max_epocas=5000, tolerancia=0.001)
    
    plt.plot(erros, label=f'Taxa (LR) = {lr}')
    resultados_lr.append({'Taxa de Aprendizagem': lr, 'Épocas até Convergência': epocas, 'Erro Final': round(erros[-1], 6)})

plt.title("Convergência do Erro variando a Taxa de Aprendizagem (LR)")
plt.xlabel("Épocas")
plt.ylabel("Erro Quadrático Médio")
plt.legend()
plt.grid(True)
plt.savefig("grafico_taxa_aprendizagem.png")
print("\n✅ Gráfico 'grafico_taxa_aprendizagem.png' salvo com sucesso!")

df_lr = pd.DataFrame(resultados_lr)
print("\nTabela 2: Influência da Taxa de Aprendizagem (XOR)")
print(df_lr.to_markdown(index=False))