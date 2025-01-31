# 31-01-2025

# 

import requests
import torch
from torchvision import datasets, transforms
import random
import io

# URL da API de previsão (troque pelo endereço correto do servidor)
API_URL = "http://meuservidor.com/predict"

# Carregar o conjunto de validação MNIST
val_dataset = datasets.MNIST(
    root=".",  # Diretório onde o dataset será salvo
    train=False,  # Define que é o conjunto de validação
    download=True,  # Baixa o dataset caso não exista
    transform=transforms.ToTensor()  # Converte as imagens para tensores
)

# Escolher uma imagem aleatória do dataset
idx = random.randint(0, len(val_dataset) - 1)
image, label = val_dataset[idx]  # Pegamos a imagem e seu rótulo verdadeiro

# Converter o tensor da imagem para bytes (necessário para enviar via API)
image_bytes = io.BytesIO()
torch.save(image, image_bytes)  # Salvar como bytes
image_bytes.seek(0)  # Posicionar no início do arquivo de bytes

# Enviar solicitação para a API com a imagem
files = {"file": image_bytes.getvalue()}  # Criar o payload da requisição
response = requests.post(API_URL, files=files)

# Exibir a resposta da API
print(f"Classe real: {label}")
print(f"Classe prevista pelo modelo: {response.json().get('class')}")



