# model/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import resnet18
import os

# Definir transformações para as imagens (tons de cinza)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Converte para tons de cinza
    transforms.Resize((224, 224)),               # Redimensiona para 224x224
    transforms.ToTensor(),                       # Converte para tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normaliza
])

# Carregar dados
dataset = datasets.ImageFolder('data', transform=transform)

# Dividir em treino e teste (80% treino, 20% teste)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Criar DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Definir a CNN (usando ResNet18 modificada para tons de cinza)
model = resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # Ajustar para 1 canal de entrada
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: osteoporose e normal

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

# Salvar o modelo treinado
torch.save(model.state_dict(), 'model/model.pth')
print("Modelo treinado e salvo em model/model.pth")
