# 30-01-2025

# Script básico para treinamento

from fastapi import FastAPI
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

# Definir o modelo (mesmo do treinamento)
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(28*28, 128)
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Inicializar API
app = FastAPI()
model = NeuralNet()
model.load_state_dict(torch.load("model/model.pth"))
model.eval()

# Transformação da imagem
transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

@app.post("/predict/")
async def predict(file: bytes):
    image = Image.open(io.BytesIO(file)).convert("L")
    image = transform(image).unsqueeze(0)  # Adicionar dimensão batch
    output = model(image)
    prediction = torch.argmax(output, dim=1).item()
    return {"prediction": prediction}