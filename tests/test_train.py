# 30-01-2025

# Script básico para treinamento

import torch
from model.train import NeuralNet

def test_model_output():
    model = NeuralNet()
    dummy_input = torch.randn(1, 28*28)  # Entrada aleatória
    output = model(dummy_input)
    assert output.shape == (1, 10), "O modelo não retorna 10 classes!"
