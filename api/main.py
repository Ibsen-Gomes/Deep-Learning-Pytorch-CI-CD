# 30-01-2025

# Script básico para treinamento

from fastapi import FastAPI  # Importa a biblioteca FastAPI para criar a API
import torch  # Importa a biblioteca PyTorch para manipulação de tensores e criação de redes neurais
import torch.nn as nn  # Importa o módulo de rede neural do PyTorch
import torchvision.transforms as transforms  # Importa transformações de dados da torchvision
from PIL import Image  # Importa a biblioteca Pillow para manipulação de imagens
import io  # Importa a biblioteca io para manipulação de fluxos de entrada/saída

# Definir o modelo (mesmo do treinamento)
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

# Inicializar API
app = FastAPI()  # Cria uma instância da aplicação FastAPI

# Carregar e preparar o modelo
model = NeuralNet()  # Instancia o modelo NeuralNet
model.load_state_dict(torch.load("model/model.pth"))  # Carrega os pesos do modelo treinado
model.eval()  # Coloca o modelo em modo de avaliação (não treinamento)

# Transformação da imagem para o formato esperado pelo modelo
transform = transforms.Compose([
    transforms.Grayscale(),  # Converte a imagem para escala de cinza
    transforms.ToTensor(),  # Converte a imagem para um tensor
    transforms.Normalize((0.5,), (0.5,))  # Normaliza a imagem
])

@app.post("/predict/")
async def predict(file: bytes):
    # Abre a imagem a partir dos bytes recebidos e converte para escala de cinza
    image = Image.open(io.BytesIO(file)).convert("L")
    # Aplica as transformações na imagem e adiciona uma dimensão de batch
    image = transform(image).unsqueeze(0)
    # Passa a imagem pelo modelo para obter a previsão
    output = model(image)
    # Obtém a classe prevista pelo modelo (índice com maior valor)
    prediction = torch.argmax(output, dim=1).item()
    # Retorna a previsão como resposta da API
    return {"prediction": prediction}
