# 30-01-2025

# 30-01-2025

# Script básico para treinamento

from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image
import torch
import io
import torch.nn as nn

app = FastAPI()

# Carregar o modelo treinado
model = torch.load('model/model.pth')
model.eval()

# Transformações para a imagem de entrada
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Ler a imagem enviada
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = transform(image).unsqueeze(0)

    # Fazer a previsão
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()

    return {"class": predicted_class}

