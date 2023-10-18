# -*- coding: utf-8 -*-

#%%
#Librerías

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
import matplotlib.pyplot as plt
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


#%%
#Visualización de datos


# Define la transformación de datos para FashionMNIST (normalización y escalado)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Descargar y cargar el conjunto de datos FashionMNIST
eval_dataset = FashionMNIST(root='./data', train=False, transform=transform, download=True)

# Elije un índice aleatorio para visualizar una imagen
random_index = random.randint(0, len(eval_dataset) - 1)

# Obtén la imagen y la etiqueta correspondiente
image, label = eval_dataset[random_index]

# Mapea el valor de la etiqueta a su descripción
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_name = class_names[label]

# Convierte la imagen en un arreglo de numpy
image = image.numpy()

# Desescala la imagen y la muestra
image = image / 2 + 0.5  # Desescalamos la imagen
plt.imshow(image[0], cmap='gray')
plt.title(f'Etiqueta: {class_name}')
plt.show()



#%%
#Carga de datos

# Definir la transformación de datos para FashionMNIST (normalización y escalado)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Descargar y cargar el conjunto de datos FashionMNIST
train_dataset = FashionMNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Descargar y cargar el conjunto de datos FashionMNIST para evaluación
eval_dataset = FashionMNIST(root='./data', train=False, transform=transform, download=True)
eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

#%%
#a)
#Diseño de la red, enternamiento y evaluación

# Definir la arquitectura de la red neuronal
#Red con una capa oculta de 32 neuronas y función de activación sigmoide
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 32)  # Capa de entrada (28x28 píxeles) a capa oculta
        self.fc2 = nn.Linear(32, 10)  # Capa oculta a capa de salida

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Aplanar la imagen
        x = torch.sigmoid(self.fc1(x))  # Aplicar sigmoide a la capa oculta
        x = self.fc2(x)  # Capa de salida
        return x

# Crear una instancia de la red neuronal
net = Net()

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

#Array para guardar las pérdidas
train_losses = []
eval_losses = []
accuracies = []

# Entrenar la red neuronal durante 10 épocas
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))
    
    # Evaluar en el conjunto de prueba
    eval_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in eval_loader:
            inputs, labels = data
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    eval_losses.append(eval_loss / len(eval_loader))
    accuracies.append(100 * correct / total)
    
    
    print(f'Época {epoch + 1}, Pérdida en entrenamiento: {train_losses[-1]}, Pérdida en evaluación: {eval_losses[-1]}, Precisión en evaluación: {100 * correct / total}%')

print('Entrenamiento completado')

# Guardar el modelo entrenado
torch.save(net.state_dict(), 'fashion_mnist_model_with_regularization.pth')

# Imprimir las listas de pérdidas en entrenamiento y evaluación
print("Pérdidas en entrenamiento:", train_losses)
print("Pérdidas en evaluación:", eval_losses)

# Crear un gráfico de las pérdidas en entrenamiento y evaluación
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Entrenamiento")
plt.plot(range(1, len(eval_losses) + 1), eval_losses, label="Evaluación")
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida en Entrenamiento y Evaluación')
plt.grid(True)
plt.show()

# Crear un gráfico con dos escalas en el eje y
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(range(1, len(train_losses) + 1), train_losses, label="Entrenamiento", color='blue')
ax1.plot(range(1, len(eval_losses) + 1), eval_losses, label="Evaluación", color='red')
ax1.set_xlabel('Época')
ax1.set_ylabel('Pérdida', color='black')

ax2 = ax1.twinx()
ax2.plot(range(1, len(accuracies) + 1), accuracies, label="Precisión en Evaluación", color='green')
ax2.set_ylabel('Precisión (%)', color='black')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="center right")

plt.title('Pérdida y Precisión en Evaluación')
plt.show()

#%%
#b.1)
#Arquitectura diferente 1
#Mas neuronas (64) en la capa oculta
# Definir la arquitectura de la red neuronal
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)  # Capa de entrada (28x28 píxeles) a capa oculta
        self.fc2 = nn.Linear(64, 10)  # Capa oculta a capa de salida

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Aplanar la imagen
        x = torch.sigmoid(self.fc1(x))  # Aplicar sigmoide a la capa oculta
        x = self.fc2(x)  # Capa de salida
        return x

# Crear una instancia de la red neuronal
net = Net()

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

train_losses = []
eval_losses = []
accuracies = []


# Entrenar la red neuronal
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))
    
    # Evaluar en el conjunto de prueba
    eval_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in eval_loader:
            inputs, labels = data
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    eval_losses.append(eval_loss / len(eval_loader))
    accuracies.append(100 * correct / total)
    
    print(f'Época {epoch + 1}, Pérdida en entrenamiento: {train_losses[-1]}, Pérdida en evaluación: {eval_losses[-1]}, Precisión en evaluación: {100 * correct / total}%')

print('Entrenamiento completado')

# Guardar el modelo entrenado
torch.save(net.state_dict(), 'fashion_mnist_model_with_regularization.pth')

# Imprimir las listas de pérdidas en entrenamiento y evaluación
print("Pérdidas en entrenamiento:", train_losses)
print("Pérdidas en evaluación:", eval_losses)

# Crear un gráfico de las pérdidas en entrenamiento y evaluación
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Entrenamiento")
plt.plot(range(1, len(eval_losses) + 1), eval_losses, label="Evaluación")
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida en Entrenamiento y Evaluación Arquitectura 1')
plt.grid(True)
plt.show()

# Crear un gráfico con dos escalas en el eje y
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(range(1, len(train_losses) + 1), train_losses, label="Entrenamiento", color='blue')
ax1.plot(range(1, len(eval_losses) + 1), eval_losses, label="Evaluación", color='red')
ax1.set_xlabel('Época')
ax1.set_ylabel('Pérdida', color='black')

ax2 = ax1.twinx()
ax2.plot(range(1, len(accuracies) + 1), accuracies, label="Precisión en Evaluación", color='green')
ax2.set_ylabel('Precisión (%)', color='black')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="center right")

plt.title('Pérdida y Precisión en Evaluación Arquitectura 1')
plt.show()



#%%
#b.2)
#Arquitectura diferente 2
#Otra capa oculta relu
# Definir la arquitectura de la red neuronal
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)  # Capa de entrada (28x28 píxeles) a capa oculta
        self.fc2 = nn.Linear(64, 32)  # Capa oculta a capa de salida
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Aplanar la imagen
        x = torch.sigmoid(self.fc1(x))  # Aplicar sigmoide a la capa oculta
        x = self.fc2(x)  # Capa oculta
        x = torch.relu(x) # Aplicar relu a la capa oculta
        x = self.fc3(x)  # Capa de salida
        return x

# Crear una instancia de la red neuronal
net = Net()

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

train_losses = []
eval_losses = []
accuracies = []

# Entrenar la red neuronal
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))
    
    # Evaluar en el conjunto de prueba
    eval_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in eval_loader:
            inputs, labels = data
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    eval_losses.append(eval_loss / len(eval_loader))
    accuracies.append(100 * correct / total)
    
    print(f'Época {epoch + 1}, Pérdida en entrenamiento: {train_losses[-1]}, Pérdida en evaluación: {eval_losses[-1]}, Precisión en evaluación: {100 * correct / total}%')

print('Entrenamiento completado')

# Guardar el modelo entrenado
torch.save(net.state_dict(), 'fashion_mnist_model_with_regularization.pth')

# Imprimir las listas de pérdidas en entrenamiento y evaluación
print("Pérdidas en entrenamiento:", train_losses)
print("Pérdidas en evaluación:", eval_losses)

# Crear un gráfico de las pérdidas en entrenamiento y evaluación
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Entrenamiento")
plt.plot(range(1, len(eval_losses) + 1), eval_losses, label="Evaluación")
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida en Entrenamiento y Evaluación Arquitectura 2')
plt.grid(True)
plt.show()

# Crear un gráfico con dos escalas en el eje y
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(range(1, len(train_losses) + 1), train_losses, label="Entrenamiento", color='blue')
ax1.plot(range(1, len(eval_losses) + 1), eval_losses, label="Evaluación", color='red')
ax1.set_xlabel('Época')
ax1.set_ylabel('Pérdida', color='black')

ax2 = ax1.twinx()
ax2.plot(range(1, len(accuracies) + 1), accuracies, label="Precisión en Evaluación", color='green')
ax2.set_ylabel('Precisión (%)', color='black')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="center right")

plt.title('Pérdida y Precisión en Evaluación Arquitectura 2')
plt.show()

#%%
#b.3)
#Arquitectura diferente 3
#Capa oculta 128 relu, capa oculta 32 tanh
# Definir la arquitectura de la red neuronal
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Capa de entrada (28x28 píxeles) a capa oculta
        self.fc2 = nn.Linear(128, 32)  # Capa oculta a capa de salida
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Aplanar la imagen
        x = torch.relu(self.fc1(x))  # Aplicar relu a la capa oculta
        x = self.fc2(x)  # Capa oculta
        x = torch.tanh(x) # Aplicar tanh a la capa oculta
        x = self.fc3(x)  # Capa de salida
        return x

# Crear una instancia de la red neuronal
net = Net()

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

train_losses = []
eval_losses = []
accuracies = []

# Entrenar la red neuronal
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))
    
    # Evaluar en el conjunto de prueba
    eval_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in eval_loader:
            inputs, labels = data
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    eval_losses.append(eval_loss / len(eval_loader))
    accuracies.append(100 * correct / total)
    
    print(f'Época {epoch + 1}, Pérdida en entrenamiento: {train_losses[-1]}, Pérdida en evaluación: {eval_losses[-1]}, Precisión en evaluación: {100 * correct / total}%')

print('Entrenamiento completado')

# Guardar el modelo entrenado
torch.save(net.state_dict(), 'fashion_mnist_model_with_regularization.pth')

# Imprimir las listas de pérdidas en entrenamiento y evaluación
print("Pérdidas en entrenamiento:", train_losses)
print("Pérdidas en evaluación:", eval_losses)

# Crear un gráfico de las pérdidas en entrenamiento y evaluación
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Entrenamiento")
plt.plot(range(1, len(eval_losses) + 1), eval_losses, label="Evaluación")
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida en Entrenamiento y Evaluación Arquitectura 3')
plt.grid(True)
plt.show()

# Crear un gráfico con dos escalas en el eje y
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(range(1, len(train_losses) + 1), train_losses, label="Entrenamiento", color='blue')
ax1.plot(range(1, len(eval_losses) + 1), eval_losses, label="Evaluación", color='red')
ax1.set_xlabel('Época')
ax1.set_ylabel('Pérdida', color='black')

ax2 = ax1.twinx()
ax2.plot(range(1, len(accuracies) + 1), accuracies, label="Precisión en Evaluación", color='green')
ax2.set_ylabel('Precisión (%)', color='black')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="center right")

plt.title('Pérdida y Precisión en Evaluación Arquitectura 3')
plt.show()


#%%
#c)
#Regularización

# Definir la arquitectura de la red neuronal
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Capa de entrada (28x28 píxeles) a capa oculta
        self.fc2 = nn.Linear(128, 32)  # Capa oculta a capa de salida
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Aplanar la imagen
        x = torch.relu(self.fc1(x))  # Aplicar relu a la capa oculta
        x = self.fc2(x)  # Capa oculta
        x = torch.tanh(x) # Aplicar tanh a la capa oculta
        x = self.fc3(x)  # Capa de salida
        return x

# Crear una instancia de la red neuronal
net = Net()

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()

# Agregar regularización L2 a la función de costo
weight_decay = 0.001
optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=weight_decay)

# Listas para almacenar los valores de pérdida en cada época
train_losses = []
eval_losses = []
accuracies = []

# Entrenar la red neuronal
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        
        # Agregar término de regularización L2 a la pérdida
        l2_reg = torch.tensor(0.)
        for param in net.parameters():
            l2_reg += torch.norm(param)
        loss = loss + weight_decay * l2_reg

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))
    
    # Evaluar en el conjunto de prueba
    eval_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in eval_loader:
            inputs, labels = data
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    eval_losses.append(eval_loss / len(eval_loader))
    accuracies.append(100 * correct / total)
    
    print(f'Época {epoch + 1}, Pérdida en entrenamiento: {train_losses[-1]}, Pérdida en evaluación: {eval_losses[-1]}, Precisión en evaluación: {100 * correct / total}%')

print('Entrenamiento completado')

# Guardar el modelo entrenado
torch.save(net.state_dict(), 'fashion_mnist_model_with_regularization.pth')

# Imprimir las listas de pérdidas en entrenamiento y evaluación
print("Pérdidas en entrenamiento:", train_losses)
print("Pérdidas en evaluación:", eval_losses)

# Crear un gráfico de las pérdidas en entrenamiento y evaluación
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label="Entrenamiento")
plt.plot(range(1, len(eval_losses) + 1), eval_losses, label="Evaluación")
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.title('Pérdida en Entrenamiento y Evaluación - Regularización')
plt.grid(True)
plt.show()

# Crear un gráfico con dos escalas en el eje y
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(range(1, len(train_losses) + 1), train_losses, label="Entrenamiento", color='blue')
ax1.plot(range(1, len(eval_losses) + 1), eval_losses, label="Evaluación", color='red')
ax1.set_xlabel('Época')
ax1.set_ylabel('Pérdida', color='black')

ax2 = ax1.twinx()
ax2.plot(range(1, len(accuracies) + 1), accuracies, label="Precisión en Evaluación", color='green')
ax2.set_ylabel('Precisión (%)', color='black')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="center right")

plt.title('Pérdida y Precisión en Evaluación - Regularización')
plt.show()

#%%
#d)
#Evalúe su performance sobre el conjunto de evaluación utilizando
# accuracy, precision, recall y medida F1 para cada una de las clases
#Construya la matriz de confusión


# Cambiar la red al modo de evaluación
net.eval()  

true_labels = []
predicted_labels = []

# Evaluar en el conjunto de prueba
with torch.no_grad():
    for data in eval_loader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        
        true_labels.extend(labels.numpy())
        predicted_labels.extend(predicted.numpy())

# Calcular las métricas
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average=None)
recall = recall_score(true_labels, predicted_labels, average=None)
f1 = f1_score(true_labels, predicted_labels, average=None)

# Construir la matriz de confusión
confusion = confusion_matrix(true_labels, predicted_labels)

# Imprimir las métricas y la matriz de confusión
print(f'Accuracy: {accuracy}')
for i in range(10):
    print(f'Clase {i} - Precision: {precision[i]}, Recall: {recall[i]}, F1: {f1[i]}')
    
print('Matriz de Confusión:')
print(confusion)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

ConfusionMatrixDisplay.from_predictions(true_labels, predicted_labels, display_labels=class_names, xticks_rotation=90)

print(classification_report(true_labels, predicted_labels, target_names=class_names))


#%%
#e)
#Obtener las 10 instancias más difíciles usando entropía

# Cambiar la red al modo de evaluación
net.eval()  

entropies = []

# Calcular la entropía para cada instancia en el conjunto de evaluación
for i, data in enumerate(eval_loader, 0):
    inputs, _ = data
    outputs = net(inputs)
    softmax = torch.nn.functional.softmax(outputs, dim=1)
    log_softmax = torch.log(softmax)
    entropy = -torch.sum(softmax * log_softmax, dim=1).detach().numpy()
    entropies.append(entropy)

entropies = np.array(entropies)
difficult_samples_indices = np.argsort(entropies, axis=0)[-10:]  # Obtener las 10 muestras con las mayores entropías

# Imprimir los índices de las muestras más difíciles
print("Índices de las 10 muestras más difíciles:", difficult_samples_indices)

#%%
#Graficar los más dificiles

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = difficult_samples_indices[i-1][0]
    image, label = eval_dataset[sample_idx]
    image = image.numpy()
    image = image / 2 + 0.5  # Desescalamos la imagen
    figure.add_subplot(rows, cols, i)
    plt.title(class_names[label])
    plt.axis("off")
    plt.imshow(image[0], cmap='gray') 
    
plt.show()

#Graficar los más fáciles
easy_samples_indices = np.argsort(entropies, axis=0)[:10]

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = easy_samples_indices[i-1][0]
    image, label = eval_dataset[sample_idx]
    image = image.numpy()
    image = image / 2 + 0.5  # Desescalamos la imagen
    figure.add_subplot(rows, cols, i)
    plt.title(class_names[label])
    plt.axis("off")
    plt.imshow(image[0], cmap='gray') 
    
plt.show()

#Graficar aleatorios
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(eval_dataset), size=(1,)).item()
    image, label = eval_dataset[sample_idx]
    image = image.numpy()
    image = image / 2 + 0.5  # Desescalamos la imagen
    figure.add_subplot(rows, cols, i)
    plt.title(class_names[label])
    plt.axis("off")
    plt.imshow(image[0], cmap='gray') 
    
plt.show()

