from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import io

# Inicializar a API
app = FastAPI()

# ✅ Carregar o modelo MobileNetV2 corretamente
model = mobilenet_v2(weights=None)  # Tem que ser igual ao treino
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, 2)  # Duas classes: Normal e Osteoartrite

# ✅ Carregar pesos treinados
model.load_state_dict(torch.load("model/model.pth", map_location=torch.device('cpu')))
model.eval()

# Definir transformações
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

@app.post("https://deep-learning-pytorch-ci-cd-1.onrender.com/predict/")
async def predict(file: UploadFile = File(...)):
    """Recebe uma imagem e retorna a previsão"""
    try:
        # Ler a imagem enviada
        image = Image.open(io.BytesIO(await file.read()))

        # Transformar a imagem para formato adequado
        image = transform(image).unsqueeze(0)  # Adiciona batch dimension

        # Fazer previsão
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()

        # Mapear saída para as classes
        classes = ["Normal", "Osteoartrite"]
        return {"prediction": classes[prediction]}

    except Exception as e:
        return {"error": str(e)}
