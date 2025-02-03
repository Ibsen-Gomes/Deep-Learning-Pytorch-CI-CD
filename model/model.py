# 30-01-2025

# Script básico para modelo

# 30-01-2025

# Script básico para modelo

# src/model.py
import torch.nn as nn

# Classe que define o modelo de Rede Neural Convolucional Simples (SimpleCNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Primeira camada convolucional: 1 canal de entrada (imagem em escala de cinza),
        # 32 canais de saída, kernel (filtro) de tamanho 3x3, stride de 1, e padding de 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Camada de pooling (subamostragem) com kernel de 2x2 e stride de 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Camada totalmente conectada: entradas vindas da camada de pooling,
        # 32*14*14 neurônios e 10 saídas correspondentes às classes do MNIST
        self.fc1 = nn.Linear(32 * 14 * 14, 10)

    # Função que define o caminho que a entrada percorre na rede
    def forward(self, x):
        # Passa a entrada pela camada convolucional e aplica ReLU
        x = self.pool(nn.functional.relu(self.conv1(x)))
        # Redimensiona (flatten) o tensor para um vetor 1D com 32*14*14 elementos
        x = x.view(-1, 32 * 14 * 14)
        # Passa o vetor pela camada totalmente conectada
        x = self.fc1(x)
        return x