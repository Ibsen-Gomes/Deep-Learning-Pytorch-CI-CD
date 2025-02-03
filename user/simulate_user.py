# 31-01-2025

# 

import requests
import torch
from torchvision import datasets, transforms
import io
from PIL import Image

# 🔹 Substitua pela URL real do Render
API_URL = "https://deep-learning-pytorch-ci-cd-1.onrender.com/predict"

# 🔹 Definir transformações para converter a imagem do MNIST
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

# 🔹 Carregar o conjunto de validação do MNIST
val_dataset = datasets.MNIST(
    root=".", train=False, download=True, transform=transform
)

def send_image(image_tensor):
    """ Envia uma imagem do MNIST para a API hospedada no Render """

    # Converter tensor para imagem PIL
    image_pil = transforms.ToPILImage()(image_tensor.squeeze(0))

    # Salvar a imagem em memória como um arquivo temporário
    img_bytes = io.BytesIO()
    image_pil.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    # Enviar a imagem para a API
    response = requests.post(API_URL, files={"file": img_bytes})
    
    return response.json()

if __name__ == "__main__":
    # Escolher uma imagem de teste do MNIST (exemplo: primeira imagem)
    image_tensor, label = val_dataset[0]  # Pegando a primeira imagem e seu rótulo verdadeiro
    
    # Fazer a previsão
    result = send_image(image_tensor)

    # Exibir a resposta da API e o rótulo real
    print(f"Resposta da API: {result}")
    print(f"Rótulo verdadeiro: {label}")




