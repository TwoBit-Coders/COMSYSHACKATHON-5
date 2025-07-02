import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import numpy as np

#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Global Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

#Macro Accuracy
def macro_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true + y_pred))
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    return np.mean(per_class_acc)


#Model Architecture
def build_model(num_classes=None, embedding=False):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    if embedding:
        model._fc = nn.Identity()
    else:
        model._fc = nn.Linear(model._fc.in_features, num_classes)
    return model.to(device)


#Training Function
def train_model(train_dir, val_dir, save_path, epochs=10):
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    val_dataset = ImageFolder(root=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"Classes: {train_dataset.classes}")

    num_classes = len(train_dataset.classes)
    model = build_model(num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        train_loss, train_preds, train_labels = 0, [], []

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        train_macro_acc = macro_accuracy(train_labels, train_preds)

        #Validation
        model.eval()
        val_loss, val_preds, val_labels = 0, [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_macro_acc = macro_accuracy(val_labels, val_preds)

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Top-1 Acc: {train_acc:.4f}, Macro Acc: {train_macro_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss/len(val_loader):.4f}, Top-1 Acc: {val_acc:.4f}, Macro Acc: {val_macro_acc:.4f}, F1: {val_f1:.4f}\n")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
