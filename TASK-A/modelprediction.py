import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#function for image processing(checking whether given image is of a male or a female)
def imagepred(imagepath,model_path):
    image_path = imagepath  

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  

    model = models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )

    model.load_state_dict(torch.load(
        model_path, 
        map_location=torch.device('cpu')
    ))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    classes = ['female', 'male']
    return predicted.item(), confidence.item(), classes[predicted.item()]

#function for test-folder processing(returning predictions,accuracy,F-1 Score,precision and recall)
def folderpred(folder_path,model_path):
    y_true = []
    y_pred = []
    predictions = []

    label_map = {'female': 0, 'male': 1}  # true label mapping

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                predicted_class_idx, conf, predicted_label = imagepred(image_path,model_path)

                # Try to get the true label from folder name
                folder_name = os.path.basename(os.path.dirname(image_path)).lower()
                true_label = label_map.get(folder_name, -1)

                if true_label != -1:
                    y_true.append(true_label)
                    y_pred.append(predicted_class_idx)

                predictions.append((file, predicted_label, conf * 100, folder_name))

    # Computing the  metrics
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred) * 100
    recall = recall_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred) * 100

    return predictions, {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
