# 30-01-2025

# Script básico para treinamento

from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image
import torch
import io
import torch.nn as nn

# Criar API FastAPI
app = FastAPI()

# 🔹 Definir a arquitetura do modelo (deve ser igual ao modelo treinado!)
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 🔹 Criar o modelo e carregar os pesos corretamente
model = NeuralNet()
model.load_state_dict(torch.load("model/model.pth", map_location=torch.device("cpu")))
model.eval()  # Agora funcionará sem erro!

# Transformação para pré-processamento das imagens
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

@app.get("/")  
def home():
    return {"message": "API de Deep Learning está rodando!"}

@app.post("https://deep-learning-pytorch-ci-cd-1.onrender.com/predict")  # 🔹 A rota precisa existir aqui!
async def predict(file: UploadFile = File(...)):
    """Recebe uma imagem e retorna a previsão do modelo"""
    
    # Ler a imagem enviada pelo usuário
    image = Image.open(io.BytesIO(await file.read()))
    image = transform(image).unsqueeze(0)  # Adicionar dimensão do batch
    
    # Fazer previsão
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()

    return {"class": predicted_class}

