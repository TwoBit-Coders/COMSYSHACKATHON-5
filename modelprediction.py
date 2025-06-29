import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def imagepred(imagepath):
    # Step 1: Loading and transforming the image
    image_path = imagepath  

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  

    # Step 2: Load the model structure (ResNet50)
    model = models.resnet50(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )

    # Step 3: Load the trained weights
    model.load_state_dict(torch.load(
        r"E:\TaskA\gender_classifier_resnet50.pth", 
        map_location=torch.device('cpu')
    ))
    model.eval()

    # Step 4: Predict the gender + confidence
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    
    classes = ['female', 'male']
    return f"Predicted Gender: {classes[predicted.item()]}, Confidence Score: {confidence.item() * 100:.2f}%"
