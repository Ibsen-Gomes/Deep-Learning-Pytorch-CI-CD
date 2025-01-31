# 31-01-2025

# 

import torch
from torchvision import datasets, transforms
import requests
import random

# URL da API hospedada no Heroku (substitua pela URL do seu app)
API_URL = "https://nome-do-seu-app.herokuapp.com/predict"

# Carregar um dado de validação
val_dataset = datasets.MNIST(".", train=False, download=True, transform=transforms.ToTensor())

# Escolher uma imagem aleatória
idx = random.randint(0, len(val_dataset) - 1)
image, label = val_dataset[idx]

# Transformar para formato esperado pelo modelo
image = image.view(-1, 28 * 28)

# Simular envio para a API (JSON)
try:
    response = requests.post(API_URL, files={"file": image.numpy().tobytes()})
    response.raise_for_status()  # Verifica se a requisição foi bem-sucedida
    print(f"Classe real: {label}")
    print(f"Classe prevista pelo modelo: {response.json().get('class')}")
except requests.exceptions.RequestException as e:
    print(f"Erro ao conectar à API: {e}")
