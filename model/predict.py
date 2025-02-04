# model/predict.py
import requests

# URL da API hospedada no Render
API_URL = "https://deep-learning-pytorch-ci-cd-1.onrender.com/predict"

# Função para enviar uma imagem e obter a previsão
def predict_image(image_path):
    try:
        # Abre a imagem no modo binário
        with open(image_path, "rb") as image_file:
            # Envia a imagem para a API
            response = requests.post(API_URL, files={"file": image_file})
            response.raise_for_status()  # Verifica se a requisição foi bem-sucedida
            return response.json()  # Retorna a resposta da API
    except requests.exceptions.RequestException as e:
        print(f"Erro ao conectar à API: {e}")
        return None

# Exemplo de uso
if __name__ == "__main__":
    # Caminho para a imagem que você quer enviar
    image_path = "validation/normal/10.png"

    # Faz a previsão
    result = predict_image(image_path)
    if result:
        print("Previsão:", result)