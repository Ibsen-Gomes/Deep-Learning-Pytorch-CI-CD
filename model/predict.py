import os
import torch
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2
from PIL import Image

# URL do repositório GitHub
REPO_URL = "https://github.com/Ibsen-Gomes/Deep-Learning-Pytorch-CI-CD.git"
CLONE_PATH = "deep-learning-model"

# Clonar repositório da branch "deploy"
if not os.path.exists(CLONE_PATH):
    print("Clonando repositório do GitHub...")
    os.system(f"git clone --branch deploy {REPO_URL} {CLONE_PATH}")

# Caminho do modelo treinado
MODEL_PATH = os.path.join(CLONE_PATH, "model", "model.pth")

# Verificar se o modelo existe no repositório clonado
if not os.path.exists(MODEL_PATH):
    print("Erro: Modelo não encontrado no repositório clonado.")
    exit()

# Carregar o modelo treinado
print("Carregando modelo...")
model = mobilenet_v2(weights=None)
num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# Definir transformações para a imagem
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Converter para 3 canais (Mobilenet espera RGB)
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Carregar uma imagem da máquina local para fazer previsões
image_path = input("Digite o caminho da imagem para análise: ")
if not os.path.exists(image_path):
    print("Erro: Arquivo de imagem não encontrado.")
    exit()

image = Image.open(image_path)
image = transform(image).unsqueeze(0)  # Adicionar batch dimension

# Fazer previsão
with torch.no_grad():
    output = model(image)
    prediction = torch.argmax(output, dim=1).item()

# Exibir resultado
classes = ["Normal", "Osteoporose"]
print(f"Previsão: {classes[prediction]}")

