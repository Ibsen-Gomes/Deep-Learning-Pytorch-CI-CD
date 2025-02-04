# Criando a API para servir o modelo (main.py)
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import CNNModel

app = FastAPI()

model = CNNModel()
model.load_state_dict(torch.load("model/model.pth", map_location=torch.device('cpu')))
model.eval()

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    image = transform_image(image_bytes)
    output = model(image)
    _, predicted = torch.max(output, 1)
    classes = ["normal", "osteoporosis"]
    return {"prediction": classes[predicted.item()]}
