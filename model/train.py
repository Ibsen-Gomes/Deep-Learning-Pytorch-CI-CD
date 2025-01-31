# 30-01-2025

# Script básico para treinamento

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Classe que define o modelo de Rede Neural Simples
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # Primeira camada linear: de 28*28 entradas para 128 saídas
        self.layer1 = nn.Linear(28*28, 128)
        # Segunda camada linear: de 128 entradas para 10 saídas (classes)
        self.layer2 = nn.Linear(128, 10)

    # Função que define o caminho que a entrada percorre na rede
    def forward(self, x):
        # Redimensiona (flatten) a entrada para um vetor de 1D com 28*28 elementos
        x = x.view(-1, 28*28)  
        # Aplica a função de ativação ReLU na primeira camada
        x = torch.relu(self.layer1(x))
        # Passa a saída da primeira camada para a segunda camada
        x = self.layer2(x)
        return x

# Prepara os dados utilizando transformações
transform = transforms.Compose([
    transforms.ToTensor(),                    # Converte imagens para tensores
    transforms.Normalize((0.5,), (0.5,))      # Normaliza os tensores
])

# Baixa e prepara o dataset MNIST para treinamento
trainset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

# Cria um DataLoader para iterar pelo dataset de treinamento
trainloader = torch.utils.data.DataLoader(
    trainset, 
    batch_size=64, 
    shuffle=True
)

# Instancia o modelo, a função de loss e o otimizador
model = NeuralNet()
criterion = nn.CrossEntropyLoss()               # Função de perda: Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Otimizador: Adam com taxa de aprendizado de 0.001

# Loop de treinamento do modelo
for epoch in range(5):  # Treina por 5 épocas
    for images, labels in trainloader:
        optimizer.zero_grad()                 # Zera os gradientes do otimizador
        outputs = model(images)               # Passa as imagens pelo modelo
        loss = criterion(outputs, labels)     # Calcula a perda comparando as saídas com os rótulos
        loss.backward()                       # Calcula os gradientes da perda
        optimizer.step()                      # Atualiza os pesos do modelo

# Salva o modelo treinado no disco
torch.save(model.state_dict(), "model/model.pth")
print("Modelo treinado e salvo!")
