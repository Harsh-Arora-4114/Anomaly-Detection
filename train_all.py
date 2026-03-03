import os
import torch
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from model import AutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

# 🔥 FIXED BASE PATH
base_path = "dataset"

# Automatically detect category folders inside dataset
categories = [folder for folder in os.listdir(base_path)
              if os.path.isdir(os.path.join(base_path, folder))]

print("Detected Categories:", categories)

epochs = 5
threshold = 0.02
results = {}

for category in categories:
    print(f"\nTraining Category: {category}")

    train_path = os.path.join(base_path, category, "train")
    test_path = os.path.join(base_path, category, "test")

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # -------- TRAIN --------
    for epoch in range(epochs):
        total_loss = 0
        for img, _ in train_loader:
            img = img.to(device)

            output = model(img)
            loss = criterion(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), f"{category}_model.pth")

    # -------- TEST --------
    print("Testing...")

    y_true = []
    y_pred = []

    test_dataset = datasets.ImageFolder(test_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        for img, label in tqdm(test_loader):
            img = img.to(device)
            output = model(img)

            error = torch.mean((output - img) ** 2).item()

            prediction = 1 if error > threshold else 0
            actual = 0 if label.item() == 0 else 1

            y_pred.append(prediction)
            y_true.append(actual)

    acc = accuracy_score(y_true, y_pred)
    results[category] = acc

    print(f"Accuracy for {category}: {acc:.4f}")

print("\nFINAL RESULTS:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")