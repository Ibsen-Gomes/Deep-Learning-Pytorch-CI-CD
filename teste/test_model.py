# 30-01-2025

# Script básico para treinamento

import torch
from src.model import SimpleCNN

def test_model_creation():
    model = SimpleCNN()
    assert model is not None, "O modelo não foi criado corretamente."

def test_model_output_shape():
    model = SimpleCNN()
    input_tensor = torch.randn(1, 1, 28, 28)  # Simula uma entrada MNIST
    output = model(input_tensor)
    assert output.shape == (1, 10), f"A saída do modelo tem forma incorreta: {output.shape}"