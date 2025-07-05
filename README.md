***👤 TASK-A: Gender Classification with ResNet50***
A plug-and-play gender classification model that predicts whether a face is male or female using a pretrained ResNet50 architecture.

    📁 Files
    🔹 gender_resnet_50.pth
    Pretrained ResNet50 model weights  fine-tuned  for gender classification.
    
    🔹 modelcreation.py
    Builds and returns the ResNet50 model with the correct structure for embedding extraction.
    
    🔹 modelprediction.py
    Contains logic to load the model and make predictions from face images.
    
    🔹 userinput.py
    Terminal-based user interface for entering image paths/folder path and displaying gender predictions.
    
    🔹 requirements.txt
    Lists all required Python libraries (e.g., torch, torchvision, PIL, etc.).

***👥 TASK-B: Face Verification with EfficientNet-B0***
A ready-to-use face verification system that uses cosine similarity between deep embeddings extracted from EfficientNet-B0.

    📁 Files
    🔹 face_recognition_model.pth
    Pretrained EfficientNet-B0 model weights fine-tuned for facial recognition.
    
    🔹 model_creation.py
    Builds and returns the EfficientNet model with the correct structure for embedding extraction.
    
    🔹 model_prediction.py
    Contains core functions to:
    • Extract face embeddings
    • Compare images using cosine similarity
    • Perform folder-based or individual image pair verification
    
    🔹 user_input.py
    Terminal-based CLI for users to:
    • Verify if two images are of the same person
    • Perform bulk verification between folders
    
    🔹 requirements.txt
    Lists all required Python libraries (e.g., torch, efficientnet_pytorch, sklearn, etc.).


--------------------------------------

## 🚀 Getting Started

### 1. Clone the repository
```bash
# Step 1: Initialize an empty Git repository
!git init

# Step 2: Clone the project repository
!git clone https://github.com/TwoBit-Coders/COMSYSHACKATHON-5.git

# Step 3: Navigate to the TASK-B or TASK-A directory
%cd /content/COMSYSHACKATHON-5/TASK-B
              OR
%cd /content/COMSYSHACKATHON-5/TASK-A

# Step 4: Install required dependencies
!pip install -r requirements.txt

# Step 5: Run the interactive face verification tool / Run the interactive Gender classification tool
!python user_input.py(for TASK-B face verification)
                OR
!python userinput.py(for TASK-A Gender classification)

# step 6: Sample Outputs
TASK-A:
    enter 1 for single image and 2 for test dataset with inner folders as male/ and female/1
    Enter the single file path: /content/Danny_Ainge_0001.jpg
    Enter the model path: /content/COMSYSHACKATHON-5/TASK-A/gender_resnet_50.pth #path of the model created in modelcreation.py
    /usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
      warnings.warn(
    /usr/local/lib/python3.11/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.
      warnings.warn(msg)
    (1, 0.8274878859519958, 'male')

TASK-B:
     Face Verification Menu
    1 :- Verify using Folder
    2 :- Verify two images
    Select an option (1/2): 2
    Enter path for first image: /content/Danny_Ainge_0001.jpg
    Enter path for second image: /content/Danny_Ainge_0001_sunny.jpg
    Enter path to your trained model (.pth file): /content/COMSYSHACKATHON-5/TASK-B/Face_recognition_model.pth ##path of the model created in model_creation.py
    Enter similarity threshold (default 0.7): 0.7
    Downloading: "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth" to /root/.cache/torch/hub/checkpoints/efficientnet-b0-355c32eb.pth
    100% 20.4M/20.4M [00:00<00:00, 164MB/s]
    Loaded pretrained weights for efficientnet-b0
    
    Cosine Similarity: 0.8313
    Same Person


----------------------------------------
modelcreation.py/model_creation.py
This script shows how the ResNet50/EfficientNet-B0 model was modified and trained. You can use it to:

Retrain the model with a new dataset

Adjust architecture or training parameters

Most users don’t need this unless you're retraining or experimenting.
------------------------------------------

🙋‍♂️ Who Should Use This?
Developers working on face-based applications

AI/ML students or researchers

Anyone looking for a plug-and-play gender classification/face verification tool

👨‍💻 Author
Made with ❤️ by ARCHISMAN BANERJEE AND ARITRA DUTTA
