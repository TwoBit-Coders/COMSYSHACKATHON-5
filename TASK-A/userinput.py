#input an image to check whether a given photo is of a male or not
from modelprediction import imagepred,folderpred
choice = int(input("Enter 1 for single image and 2 for test dataset with inner folders as male/ and female/"))
if(choice==1):
    path = input("Enter the single file path: ").strip()
    image = rf"{path}"  
    model_path = input("Enter the model path: ").strip()
    mp = rf"{model_path}"
    prediction = imagepred(image,mp)
    print(prediction)
elif(choice==2):
    path = input("Enter the test folder path: ").strip()
    image = rf"{path}"  
    model_path = input("Enter the model path: ").strip()
    mp = rf"{model_path}"
    prediction = folderpred(image,mp)
    print(prediction)
else:
    print("WRONG INPUT.........")
