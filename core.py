import numpy as np

# ── Funções de Ativação (Vetorizadas com NumPy) ─────────────────────────

def sigmoide_binaria(x):
    # np.clip previne erros de overflow matemático (valores muito grandes)
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def d_sigmoide_binaria(fx):
    return fx * (1.0 - fx)

def sigmoide_bipolar(x):
    x = np.clip(x, -500, 500)
    return (2.0 / (1.0 + np.exp(-x))) - 1.0

def d_sigmoide_bipolar(fx):
    return 0.5 * (1.0 - fx**2)

def tangente_hiperbolica(x):
    return np.tanh(x)

def d_tangente_hiperbolica(fx):
    return 1.0 - fx**2

# Mapa para uso dinâmico
FUNCOES_ATIVACAO = {
    'Sigmóide Bipolar':    (sigmoide_bipolar,    d_sigmoide_bipolar),
    'Sigmóide Binária':    (sigmoide_binaria,     d_sigmoide_binaria),
    'Tangente Hiperbólica':(tangente_hiperbolica, d_tangente_hiperbolica),
}

# ── Rede MLP Otimizada com NumPy ──────────────────────────────────────────

class MLP:
    def __init__(self, n_entrada=2, n_oculta=4, taxa_aprendizagem=0.2,
                 seed=42, nome_funcao='Sigmóide Bipolar'):
        self.lr = taxa_aprendizagem
        self.n_entrada = n_entrada
        self.n_oculta = n_oculta
        self.ativacao, self.d_ativacao = FUNCOES_ATIVACAO[nome_funcao]

        # Fixar a semente aleatória para reprodutibilidade dos resultados
        np.random.seed(seed)

        # Inicialização dos pesos e bias (viés)
        self.W1 = np.random.uniform(-1.0, 1.0, (n_entrada, n_oculta))
        self.b1 = np.random.uniform(-1.0, 1.0, (1, n_oculta))

        self.W2 = np.random.uniform(-1.0, 1.0, (n_oculta, 1))
        self.b2 = np.random.uniform(-1.0, 1.0, (1, 1))

    def forward(self, X):
        X = np.atleast_2d(X)
        
        # Propagação Camada Oculta
        self.net_h = np.dot(X, self.W1) + self.b1
        self.out_h = self.ativacao(self.net_h)
        
        # Propagação Camada de Saída
        self.net_o = np.dot(self.out_h, self.W2) + self.b2
        self.out_o = self.ativacao(self.net_o)
        
        return self.out_o

    def backward(self, X, y):
        X = np.atleast_2d(X)
        y = np.atleast_2d(y)
        
        # Erro na saída
        erro = y - self.out_o
        delta_saida = erro * self.d_ativacao(self.out_o)
        
        # Erro propagado para a camada oculta
        erro_oculto = np.dot(delta_saida, self.W2.T)
        delta_oculta = erro_oculto * self.d_ativacao(self.out_h)
        
        # Atualização dos pesos (Descida do Gradiente)
        self.W2 += self.lr * np.dot(self.out_h.T, delta_saida)
        self.b2 += self.lr * np.sum(delta_saida, axis=0, keepdims=True)
        self.W1 += self.lr * np.dot(X.T, delta_oculta)
        self.b1 += self.lr * np.sum(delta_oculta, axis=0, keepdims=True)
        
        # Erro Quadrático Médio
        return np.mean(erro**2)

    def treinar(self, X, y, max_epocas=20000, tolerancia=0.001):
        X = np.array(X)
        # Transforma o array unidimensional de y numa matriz coluna (N, 1)
        y = np.array(y).reshape(-1, 1)
        
        historico_erro = []
        for epoca in range(max_epocas):
            self.forward(X)
            erro_total = self.backward(X, y)
            historico_erro.append(erro_total)
            
            if erro_total < tolerancia:
                return epoca + 1, historico_erro

        return max_epocas, historico_erro

    def prever(self, X):
        return self.forward(np.array(X)).flatten().tolist()