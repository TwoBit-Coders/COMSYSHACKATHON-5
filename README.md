# 👤 Gender Classification using ResNet50(TASK-A)

This project provides a ready-to-use **gender classification model** built on top of ResNet50 using PyTorch. It allows users to load the pretrained model and predict gender from face images with just a few lines of code — no training required.

---

## 📦 Files in This Project

### 🔹 `gender_classifier_resnet50.pth`
Pretrained model file (PyTorch). Contains all the trained weights for gender classification.

### 🔹 `modelprediction.py`
Handles **loading the model and making predictions** on images. You can import and use this in your own project or use it with `userinput.py`.

### 🔹 `userinput.py`
Provides a **user-friendly interface** for inputting image paths and getting gender predictions in the terminal.

### 🔹 `requirements.txt`
Lists all the Python packages required to run the model and prediction scripts.

--------------------------------------

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
---------------------------------------
2. Install required libraries:
pip install -r requirements.txt
---------------------------------------
3. Run gender prediction
python userinput.py
You’ll be prompted to enter the path of an image file. The model will process it and print something like:

Predicted Gender: Male
⚙️ Optional: Train or Fine-Tune the Model
----------------------------------------
modelcreation.py
This script shows how the ResNet50 model was modified and trained. You can use it to:

Retrain the model with a new dataset

Adjust architecture or training parameters

Most users don’t need this unless you're retraining or experimenting.
------------------------------------------

🙋‍♂️ Who Should Use This?
Developers working on face-based applications

AI/ML students or researchers

Anyone looking for a plug-and-play gender classification tool

👨‍💻 Author
Made with ❤️ by ARCHISMAN BANERJEE AND ARITRA DUTTA
