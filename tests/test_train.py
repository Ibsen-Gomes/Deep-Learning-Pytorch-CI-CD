# 30-01-2025

# Script básico para teste básico das funcionalidades dos códigos

# Importa a biblioteca PyTorch para trabalhar com tensores e redes neurais
import torch

# Importa a biblioteca sys para manipular o caminho de importação de módulos
import sys

# Importa a biblioteca os para manipular caminhos de arquivos e diretórios
import os

# Adiciona o diretório raiz do projeto ao caminho de importação
# Isso permite que o Python encontre módulos em pastas superiores
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa a classe NeuralNet do módulo model.train
# Essa classe define a arquitetura da rede neural
from model.train import NeuralNet

def test_model_output():
    """
    Testa a saída do modelo NeuralNet para garantir que ele retorna o formato esperado.

    Descrição:
        Esta função cria uma instância do modelo NeuralNet, gera uma entrada aleatória
        e verifica se a saída do modelo tem o formato correto. O modelo deve retornar
        um tensor de forma (1, 10), onde 1 é o tamanho do batch e 10 é o número de classes.

    Passos:
        1. Cria uma instância do modelo NeuralNet.
        2. Gera uma entrada aleatória (dummy_input) com forma (1, 28*28), simulando uma imagem MNIST.
        3. Passa a entrada pelo modelo para obter a saída.
        4. Verifica se a saída tem a forma (1, 10), indicando que o modelo está funcionando corretamente.

    Exceções:
        Se a forma da saída não for (1, 10), a função lança um AssertionError com a mensagem:
        "O modelo não retorna 10 classes!"

    Exemplo de uso:
        test_model_output()  # Executa o teste
    """
    # Cria uma instância do modelo NeuralNet
    model = NeuralNet()

    # Gera uma entrada aleatória (dummy_input) com forma (1, 28*28)
    # Isso simula uma imagem MNIST (28x28 pixels) achatada em um vetor de 784 elementos
    dummy_input = torch.randn(1, 28*28)

    # Passa a entrada pelo modelo para obter a saída
    output = model(dummy_input)

    # Verifica se a forma da saída é (1, 10)
    # O modelo deve retornar 10 valores, correspondentes às probabilidades de cada classe
    assert output.shape == (1, 10), "O modelo não retorna 10 classes!"
