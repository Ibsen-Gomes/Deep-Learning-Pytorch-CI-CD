# Importa a classe SimpleCNN do módulo model.train
from model.train import SimpleCNN

def test_model_output():
    """
    Testa a saída do modelo SimpleCNN para garantir que ele retorna o formato esperado.
    """
    # Cria uma instância do modelo SimpleCNN
    model = SimpleCNN()

    # Gera uma entrada aleatória (dummy_input) com forma (1, 1, 28, 28)
    # Isso simula uma imagem MNIST (28x28 pixels) em tons de cinza (1 canal)
    dummy_input = torch.randn(1, 1, 28, 28)

    # Passa a entrada pelo modelo para obter a saída
    output = model(dummy_input)

    # Verifica se a forma da saída é (1, 2)
    # O modelo deve retornar 2 valores, correspondentes às probabilidades de cada classe
    assert output.shape == (1, 2), "O modelo não retorna 2 classes!"
