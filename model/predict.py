import requests
from PIL import Image
import io

def predict(image_path, api_url):
    # Abre a imagem e converte para bytes
    with open(image_path, 'rb') as image_file:
        image_bytes = image_file.read()

    # Envia a imagem para a API no Render
    response = requests.post(api_url, files={'file': image_bytes})

    if response.status_code == 200:
        prediction = response.json().get('prediction', 'Erro na previsão')
        return prediction
    else:
        return f"Erro ao acessar a API: {response.status_code}"

if __name__ == '__main__':
    # Substitua pela URL da sua API no Render
    api_url = 'https://seu-servico-render.onrender.com/predict/'
    
    # Substitua pelo caminho da imagem que você deseja prever
    image_path = 'path_to_your_image.jpg'
    
    print(f'Prediction: {predict(image_path, api_url)}')