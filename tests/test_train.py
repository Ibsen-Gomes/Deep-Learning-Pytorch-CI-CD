# tests/test_train.py
import torch
import sys
import os

# Adiciona o diretório raiz do projeto ao caminho de importação
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa a função create_model do módulo model.train
from model.train import create_model

def test_model_output():
    """
    Testa a saída do modelo para garantir que ele retorna o formato esperado.
    """
    # Cria uma instância do modelo
    model = create_model()

    # Gera uma entrada aleatória (dummy_input) com forma (1, 1, 224, 224)
    # Isso simula uma imagem em tons de cinza (1 canal) com tamanho 224x224
    dummy_input = torch.randn(1, 1, 224, 224)

    # Passa a entrada pelo modelo para obter a saída
    output = model(dummy_input)

    # Verifica se a forma da saída é (1, 2)
    # O modelo deve retornar 2 valores, correspondentes às probabilidades de cada classe
    assert output.shape == (1, 2), "O modelo não retorna 2 classes!"