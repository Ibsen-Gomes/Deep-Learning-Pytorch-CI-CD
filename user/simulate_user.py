# 31-01-2025

# 

import torch  # Importa a biblioteca PyTorch para manipulação de tensores e criação de redes neurais
from torchvision import datasets, transforms  # Importa módulos do torchvision para manipulação de datasets e transformação de dados
import requests  # Importa a biblioteca requests para fazer requisições HTTP
import random  # Importa a biblioteca random para geração de números aleatórios

# URL da API hospedada no Heroku (substitua pela URL do seu app)
API_URL = "https://nome-do-seu-app.herokuapp.com/predict"

# Carregar um dado de validação
val_dataset = datasets.MNIST(
    ".",  # Diretório onde o dataset será salvo
    train=False,  # Define que é o conjunto de dados de validação (não de treinamento)
    download=True,  # Baixa o dataset se não estiver presente no diretório
    transform=transforms.ToTensor()  # Transforma as imagens em tensores
)

# Escolher uma imagem aleatória
idx = random.randint(0, len(val_dataset) - 1)  # Gera um índice aleatório entre 0 e o tamanho do dataset de validação
image, label = val_dataset[idx]  # Seleciona a imagem e o rótulo correspondente no índice aleatório

# Transformar para formato esperado pelo modelo
image = image.view(-1, 28 * 28)  # Achata a imagem para um vetor 1D com 28*28 elementos

# Simular envio para a API (JSON)
try:
    # Faz uma requisição POST para a URL da API com a imagem transformada em bytes
    response = requests.post(
        API_URL,  # URL da API para onde a requisição será enviada
        files={"file": image.numpy().tobytes()}  # Converte a imagem para bytes e a envia como um arquivo
    )
    response.raise_for_status()  # Verifica se a requisição foi bem-sucedida (código de status 200)
    
    # Imprime a classe real da imagem e a classe prevista pelo modelo, que é recebida da resposta da API
    print(f"Classe real: {label}")
    print(f"Classe prevista pelo modelo: {response.json().get('class')}")
except requests.exceptions.RequestException as e:
    # Captura exceções relacionadas à requisição e imprime uma mensagem de erro
    print(f"Erro ao conectar à API: {e}")


