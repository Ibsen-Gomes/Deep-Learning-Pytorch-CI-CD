# api/main.py
from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()

# Carregar o modelo treinado
model = torch.load('model/model.pth', map_location=torch.device('cpu'))
model.eval()

# Transformações para a imagem de entrada
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

@app.post("https://deep-learning-pytorch-ci-cd-1.onrender.com/predict")
async def predict(file: UploadFile = File(...)):
    # Ler a imagem enviada
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L")  # Converte para tons de cinza
    image = transform(image).unsqueeze(0)

    # Fazer a previsão
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = "osteoporosis" if predicted.item() == 0 else "normal"

    return {"class": predicted_class}
