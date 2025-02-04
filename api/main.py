from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms
import io

app = FastAPI()

# Carrega o modelo
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Carrega o modelo treinado
model = SimpleCNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Lê a imagem enviada
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('L')  # Converte para tons de cinza

    # Pré-processa a imagem
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)

    # Faz a previsão
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return {"prediction": 'Osteoporosis' if predicted.item() == 1 else 'Normal'}
