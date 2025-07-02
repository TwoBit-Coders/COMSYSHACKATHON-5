import os
import torch
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from model_creation import build_model, device

#Global Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


#Get Embedding for a Single Image
def get_embedding(model, img_path):
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img)
    return embedding.cpu().numpy()


#Verify Images in Folder
def verify_folder(data_dir, model_path, threshold=0.7):
    model = build_model(embedding=True)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    # Build Gallery
    gallery = {}
    for person in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person)
        if os.path.isdir(person_path):
            gallery[person] = []
            for img in os.listdir(person_path):
                img_path = os.path.join(person_path, img)
                if os.path.isfile(img_path) and not img.lower().endswith(('db', 'txt')):
                    gallery[person].append(get_embedding(model, img_path))

    print(f"\nGallery prepared with {len(gallery)} persons.")

    y_true = []
    y_pred = []

    for person in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person)
        distortion_path = os.path.join(person_path, 'distortion')
        if not os.path.isdir(distortion_path):
            continue

        for img in os.listdir(distortion_path):
            img_path = os.path.join(distortion_path, img)
            if not img.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue

            embed_query = get_embedding(model, img_path)

            similarities = {}
            for gal_person, embeds in gallery.items():
                sims = [cosine_similarity(embed_query, e)[0][0] for e in embeds]
                similarities[gal_person] = np.max(sims)

            pred_person = max(similarities, key=similarities.get)
            top_score = similarities[pred_person]

            y_true.append(person)
            y_pred.append(pred_person)

            decision = "Same " if (pred_person == person and top_score > threshold) else "Different"
            print(f"\nQuery: {img}")
            print(f"Matched with: {pred_person} (Similarity: {top_score:.4f}) → {decision}")

    # Metrics
    top1_acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(gallery.keys()))
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    macro_acc = np.mean(per_class_acc)

    print("\n Verification Results:")
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Macro Average Accuracy: {macro_acc:.4f}")


#Verify Two Images
def verify_two_images(model_path, img1_path, img2_path, threshold=0.7):
    model = build_model(embedding=True)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()

    embed1 = get_embedding(model, img1_path)
    embed2 = get_embedding(model, img2_path)

    similarity = cosine_similarity(embed1, embed2)[0][0]
    print(f"\nCosine Similarity: {similarity:.4f}")

    if similarity > threshold:
        print("Same Person")
    else:
        print("Different Person")
