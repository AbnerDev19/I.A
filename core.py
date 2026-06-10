import numpy as np

def sigmoide_bipolar(x):
    return (2 / (1 + np.exp(-x))) - 1

def derivada_sigmoide_bipolar(x):
    sig = sigmoide_bipolar(x)
    return 0.5 * (1 + sig) * (1 - sig)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.lr = learning_rate
        self.hidden_size = hidden_size
        
        # Inicialização randômica dos pesos entre -1 e 1
        self.W1 = np.random.uniform(-1, 1, (input_size, self.hidden_size))
        self.b1 = np.random.uniform(-1, 1, (1, self.hidden_size))
        self.W2 = np.random.uniform(-1, 1, (self.hidden_size, output_size))
        self.b2 = np.random.uniform(-1, 1, (1, output_size))

    def forward(self, X):
        self.z_in = np.dot(X, self.W1) + self.b1
        self.z = sigmoide_bipolar(self.z_in)
        
        self.y_in = np.dot(self.z, self.W2) + self.b2
        self.y_out = sigmoide_bipolar(self.y_in)
        
        return self.y_out

    def backward(self, X, y_true):
        # Cálculo do erro na saída
        error = y_true - self.y_out
        delta_out = error * derivada_sigmoide_bipolar(self.y_in)

        # Propagação do erro para a camada intermediária
        error_hidden = delta_out.dot(self.W2.T)
        delta_hidden = error_hidden * derivada_sigmoide_bipolar(self.z_in)

        # Atualização dos pesos e bias
        self.W2 += self.z.T.dot(delta_out) * self.lr
        self.b2 += np.sum(delta_out, axis=0, keepdims=True) * self.lr
        
        # É necessário remodelar X para a multiplicação de matrizes correta
        X_reshaped = X.reshape(1, -1) 
        self.W1 += X_reshaped.T.dot(delta_hidden) * self.lr
        self.b1 += np.sum(delta_hidden, axis=0, keepdims=True) * self.lr

        return np.mean(np.abs(error))

    def treinar(self, X_train, y_train, max_epocas, erro_tolerado):
        historico_erros = []
        epoca_atual = 0
        
        for epoca in range(max_epocas):
            erro_total = 0
            # Treinamento padrão por padrão (online)
            for i in range(len(X_train)):
                saida = self.forward(X_train[i])
                erro = self.backward(X_train[i], y_train[i:i+1])
                erro_total += erro
                
            erro_medio_epoca = erro_total / len(X_train)
            historico_erros.append(erro_medio_epoca)
            epoca_atual = epoca + 1
            
            if erro_medio_epoca <= erro_tolerado:
                break
                
        return historico_erros, epoca_atual