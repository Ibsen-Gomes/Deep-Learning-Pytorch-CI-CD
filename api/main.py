# 30-01-2025

# Script básico para treinamento

from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image
import torch
import io

# Criar API FastAPI
app = FastAPI()

# Carregar o modelo treinado
model = torch.load("model/model.pth", map_location=torch.device("cpu"))
model.eval()

# Transformação da imagem
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

@app.get("/")  
def home():
    return {"message": "API de Deep Learning está rodando!"}

@app.post("/predict/")  # 🔹 A rota precisa existir aqui!
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

