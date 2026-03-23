# ===============================
# IMPORTS
# ===============================
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.utils as vutils

import numpy as np
import time

from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.metrics import confusion_matrix

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# PART A — CNN (CIFAR-10)
# ===============================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

trainset, valset = random_split(dataset, [train_size, val_size])
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64)

# -------------------------------
# CNN Model
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(128*4*4,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,10)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)

# -------------------------------
# Training Function
# -------------------------------
def train_model(model, epochs=3):

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):

        total_loss = 0

        for images, labels in trainloader:

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(trainloader):.4f}")

    return model

# -------------------------------
# Train CNN
# -------------------------------
print("\nTraining Custom CNN...")
cnn_model = train_model(SimpleCNN())

# -------------------------------
# Transfer Learning (ResNet18)
# -------------------------------
print("\nTraining ResNet18...")
resnet = models.resnet18(pretrained=True)

for param in resnet.parameters():
    param.requires_grad = False

resnet.fc = nn.Linear(resnet.fc.in_features, 10)
resnet = train_model(resnet)

# -------------------------------
# Evaluation
# -------------------------------
def evaluate(model):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, pred = torch.max(outputs, 1)

            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

cnn_acc = evaluate(cnn_model)
resnet_acc = evaluate(resnet)

print("\n=== CNN RESULTS ===")
print("Custom CNN Accuracy:", cnn_acc)
print("ResNet18 Accuracy:", resnet_acc)

# ===============================
# PART B — RNN / LSTM / GRU
# ===============================

print("\nLoading IMDB dataset...")

max_features = 10000
max_len = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)

X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# -------------------------------
# RNN Model
# -------------------------------
class TextModel(nn.Module):
    def __init__(self, model_type):
        super().__init__()

        self.embedding = nn.Embedding(max_features, 64)

        if model_type == "RNN":
            self.rnn = nn.RNN(64,64,batch_first=True)
        elif model_type == "LSTM":
            self.rnn = nn.LSTM(64,64,batch_first=True)
        else:
            self.rnn = nn.GRU(64,64,batch_first=True)

        self.fc = nn.Linear(64,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.embedding(x)
        out,_ = self.rnn(x)
        out = out[:,-1,:]
        return self.sigmoid(self.fc(out))

# -------------------------------
# Train Text Model
# -------------------------------
def train_text(model_type):

    model = TextModel(model_type).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):

        total_loss = 0

        for Xb, yb in train_loader:

            Xb = Xb.to(device)
            yb = yb.float().to(device)

            outputs = model(Xb).squeeze()
            loss = criterion(outputs, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"{model_type} Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    return model

# -------------------------------
# Train Models
# -------------------------------
print("\nTraining RNN...")
rnn_model = train_text("RNN")

print("\nTraining LSTM...")
lstm_model = train_text("LSTM")

print("\nTraining GRU...")
gru_model = train_text("GRU")

# -------------------------------
# Evaluate Text Models
# -------------------------------
def evaluate_text(model):

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for Xb, yb in test_loader:

            Xb = Xb.to(device)
            yb = yb.to(device)

            outputs = model(Xb).squeeze()
            preds = (outputs > 0.5).int()

            correct += (preds == yb).sum().item()
            total += yb.size(0)

    return 100 * correct / total

print("\n=== RNN RESULTS ===")
print("RNN Accuracy:", evaluate_text(rnn_model))
print("LSTM Accuracy:", evaluate_text(lstm_model))
print("GRU Accuracy:", evaluate_text(gru_model))

# ===============================
# PART C — GAN
# ===============================

print("\nTraining GAN...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

latent_dim = 100

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,784),
            nn.Tanh()
        )

    def forward(self,z):
        return self.model(z).view(-1,1,28,28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784,512),
            nn.LeakyReLU(0.2),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,img):
        img = img.view(img.size(0),-1)
        return self.model(img)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
opt_G = optim.Adam(G.parameters(), lr=0.0002)
opt_D = optim.Adam(D.parameters(), lr=0.0002)

for epoch in range(5):

    for imgs,_ in loader:

        real = imgs.to(device)
        batch = real.size(0)

        real_labels = torch.ones(batch,1).to(device)
        fake_labels = torch.zeros(batch,1).to(device)

        # Train D
        opt_D.zero_grad()

        real_loss = criterion(D(real), real_labels)

        z = torch.randn(batch, latent_dim).to(device)
        fake = G(z)

        fake_loss = criterion(D(fake.detach()), fake_labels)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        opt_D.step()

        # Train G
        opt_G.zero_grad()

        g_loss = criterion(D(fake), real_labels)

        g_loss.backward()
        opt_G.step()

    print(f"Epoch {epoch+1} D:{d_loss.item():.4f} G:{g_loss.item():.4f}")

    vutils.save_image(fake[:25], f"epoch_{epoch+1}.png", nrow=5, normalize=True)

print("\nGAN training complete. Images saved.")