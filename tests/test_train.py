# 30-01-2025

# Script básico para treinamento

import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.train import NeuralNet

def test_model_output():
    model = NeuralNet()
    dummy_input = torch.randn(1, 28*28)  # Entrada aleatória
    output = model(dummy_input)
    assert output.shape == (1, 10), "O modelo não retorna 10 classes!"
