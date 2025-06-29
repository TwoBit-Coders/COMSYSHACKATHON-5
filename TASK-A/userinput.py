#input an image to check whether a given photo is of a male or not
from modelprediction import imagepred

path = input("Enter the file path: ").strip()
image = rf"{path}"  
prediction = imagepred(image)
print(prediction)
