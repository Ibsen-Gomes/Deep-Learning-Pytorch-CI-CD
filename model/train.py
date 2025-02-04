import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2  # ✅ Importando MobileNetV2
import os

# Definir transformações para imagens em tons de cinza
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Converte para escala de cinza
    transforms.Resize((224, 224)),  # Redimensiona para o tamanho adequado
    transforms.ToTensor(),  # Converte para tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normaliza os dados
])

# Carregar os dados das pastas 'osteoporosis' e 'normal'
dataset = datasets.ImageFolder('data', transform=transform)

# Dividir o dataset em treino (80%) e teste (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Criar DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ✅ Modificando para usar MobileNetV2
model = mobilenet_v2(weights=None)  # ❌ Remove pretrained para economizar memória
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, 2)  # 2 classes: Osteoartrite e Normal

# Definir loss e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinar o modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(10):  # 10 épocas
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# ✅ Salvar o modelo treinado
torch.save(model.state_dict(), 'model/model.pth')
print("Modelo treinado e salvo em model/model.pth")
