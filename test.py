import torch
import cv2
import numpy as np
from torchvision import transforms
from model import AutoEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoEncoder().to(device)
model.load_state_dict(torch.load("autoencoder.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

image_path = "dataset/bottle/test/broken_large/000.png"

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256,256))
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)

error = torch.mean((output - img_tensor) ** 2).item()

print("Reconstruction Error:", error)

if error > 0.02:
    print("Anomaly Detected!")
else:
    print("Normal Image")