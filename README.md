👤 TASK-A: Gender Classification with ResNet50
A plug-and-play gender classification model that predicts whether a face is male or female using a pretrained ResNet50 architecture.

📁 Files
gender_classifier_resnet50.pth
Pretrained ResNet50 model weights for gender classification.

modelprediction.py
Contains logic to load the model and make predictions from face images.

userinput.py
Terminal-based user interface for entering image paths and displaying gender predictions.

requirements.txt
Lists all required Python libraries (e.g., torch, torchvision, PIL, etc.).

👥 TASK-B: Face Verification with EfficientNet-B0
A ready-to-use face verification system that uses cosine similarity between deep embeddings extracted from EfficientNet-B0.

📁 Files
face_recognition_model.pth
Pretrained EfficientNet-B0 model weights fine-tuned for facial recognition.

model_creation.py
Builds and returns the EfficientNet model with the correct structure for embedding extraction.

model_prediction.py
Contains core functions to:

Extract face embeddings

Compare images using cosine similarity

Perform folder-based or individual image pair verification

user_input.py
Terminal-based CLI for users to:

Verify if two images are of the same person

Perform bulk verification between folders

requirements.txt
Lists all required Python libraries (e.g., torch, efficientnet_pytorch, sklearn, etc.)


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
