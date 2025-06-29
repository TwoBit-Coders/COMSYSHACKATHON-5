# STEP 1: Imports and Setup
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# STEP 2: Transforms (Resize, Tensor, Normalize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # Mean for ImageNet
                         [0.229, 0.224, 0.225])  # Std for ImageNet
])

# STEP 3: Loading the  Dataset
train_path = "/content/hackathon/Comys_Hackathon5/Task_A/train"
val_path = "/content/hackathon/Comys_Hackathon5/Task_A/val"

train_dataset = datasets.ImageFolder(train_path, transform=transform)
val_dataset = datasets.ImageFolder(val_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

print(f"Classes: {train_dataset.classes}")  # ['female', 'male']

# STEP 4: Load Pretrained ResNet18 and Modify
model = models.resnet50(pretrained=True)

# Freeze base layers (optional)
for param in model.parameters():
    param.requires_grad = False  # Freeze everything except classifier

# Modify final FC layer for binary classification
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 2)  # Binary output (female, male)
)

model = model.to(device)

# STEP 5: Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  

# STEP 6: Training Loop
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total * 100
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

# STEP 7: Validation
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

val_acc = correct / total * 100
print(f"\nValidation Accuracy: {val_acc:.2f}%")

# STEP 8: Saving Model 
torch.save(model.state_dict(), "gender_classifier_resnet50.pth")

