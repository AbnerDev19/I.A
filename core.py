# Constante de Euler para substituir a biblioteca math
E = 2.718281828459045

class GeradorAleatorio:
    def __init__(self, seed=42):
        self.estado = seed

    def aleatorio(self):
        a = 1103515245
        c = 12345
        m = 2**31
        self.estado = (a * self.estado + c) % m
        return self.estado / m

    def uniform(self, min_val, max_val):
        return min_val + (max_val - min_val) * self.aleatorio()

# ── Funções de Ativação ────────────────────────────────────────────────────────

def sigmoide_binaria(x):
    """f(x) = 1 / (1 + e^-x)  →  saída em (0, 1)"""
    if x < -50: return 0.0
    if x > 50:  return 1.0
    return 1.0 / (1.0 + (E ** -x))

def d_sigmoide_binaria(fx):
    """f'(x) = f(x) * (1 - f(x))"""
    return fx * (1.0 - fx)

def sigmoide_bipolar(x):
    """f(x) = 2/(1 + e^-x) - 1  →  saída em (-1, 1)"""
    if x < -50: return -1.0
    if x > 50:  return  1.0
    return (2.0 / (1.0 + (E ** -x))) - 1.0

def d_sigmoide_bipolar(fx):
    """f'(x) = 0.5 * (1 - f(x)^2)"""
    return 0.5 * (1.0 - fx * fx)

def tangente_hiperbolica(x):
    """tanh(x) = (e^x - e^-x) / (e^x + e^-x)  →  saída em (-1, 1)"""
    if x < -50: return -1.0
    if x > 50:  return  1.0
    ep = E **  x
    en = E ** -x
    return (ep - en) / (ep + en)

def d_tangente_hiperbolica(fx):
    """tanh'(x) = 1 - tanh(x)^2"""
    return 1.0 - fx * fx

# Mapa para uso dinâmico
FUNCOES_ATIVACAO = {
    'Sigmóide Bipolar':    (sigmoide_bipolar,    d_sigmoide_bipolar),
    'Sigmóide Binária':    (sigmoide_binaria,     d_sigmoide_binaria),
    'Tangente Hiperbólica':(tangente_hiperbolica, d_tangente_hiperbolica),
}

# ── Rede MLP ──────────────────────────────────────────────────────────────────

class MLP_Zero_Bibliotecas:
    def __init__(self, n_entrada=2, n_oculta=4, taxa_aprendizagem=0.2,
                 seed=42, nome_funcao='Sigmóide Bipolar'):
        self.lr = taxa_aprendizagem
        self.n_entrada = n_entrada
        self.n_oculta  = n_oculta
        self.ativacao, self.d_ativacao = FUNCOES_ATIVACAO[nome_funcao]

        rng = GeradorAleatorio(seed)

        self.W1 = []
        for _ in range(n_entrada + 1):       # +1 para bias
            linha = [rng.uniform(-1.0, 1.0) for _ in range(n_oculta)]
            self.W1.append(linha)

        self.W2 = []
        for _ in range(n_oculta + 1):        # +1 para bias
            self.W2.append([rng.uniform(-1.0, 1.0)])

    def forward(self, x):
        x_bias = [1.0] + x

        saida_oculta = []
        for j in range(self.n_oculta):
            net = sum(x_bias[i] * self.W1[i][j] for i in range(len(x_bias)))
            saida_oculta.append(self.ativacao(net))

        h_bias = [1.0] + saida_oculta

        net_saida = sum(h_bias[i] * self.W2[i][0] for i in range(len(h_bias)))
        saida_final = self.ativacao(net_saida)

        return x_bias, saida_oculta, h_bias, saida_final

    def treinar(self, X, y, max_epocas=20000, tolerancia=0.001):
        historico_erro = []
        for epoca in range(max_epocas):
            erro_total = 0.0

            for index in range(len(X)):
                xi = X[index]
                yi = y[index]

                x_bias, saida_oculta, h_bias, saida_final = self.forward(xi)

                erro = yi - saida_final
                erro_total += 0.5 * (erro * erro)

                # Fase Backward ─ camada de saída
                delta_saida = erro * self.d_ativacao(saida_final)

                # Fase Backward ─ camada oculta
                delta_oculta = []
                for j in range(self.n_oculta):
                    peso_w2 = self.W2[j + 1][0]   # pula o bias (índice 0)
                    delta = self.d_ativacao(saida_oculta[j]) * peso_w2 * delta_saida
                    delta_oculta.append(delta)

                # Atualização W2
                for i in range(len(h_bias)):
                    self.W2[i][0] += self.lr * h_bias[i] * delta_saida

                # Atualização W1
                for i in range(len(x_bias)):
                    for j in range(self.n_oculta):
                        self.W1[i][j] += self.lr * x_bias[i] * delta_oculta[j]

            historico_erro.append(erro_total)
            if erro_total < tolerancia:
                return epoca + 1, historico_erro

        return max_epocas, historico_erro

    def prever(self, X):
        return [self.forward(xi)[3] for xi in X]
