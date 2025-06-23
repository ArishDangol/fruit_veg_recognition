# import sys
# import os
# import torch
# import torchvision.transforms as transforms
# from PIL import Image

# # ✅ Fix Import Issue
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from models.custom_cnn import FruitVegCNN

# # ✅ Load Model
# model = FruitVegCNN(num_classes=131)
# model.load_state_dict(torch.load('./saved_models/best_model.pth', map_location=torch.device('cpu')))
# model.eval()

# # ✅ Function to Predict Image
# def predict_image(image_path):
#     # ✅ Check if Image Exists
#     if not os.path.exists(image_path):
#         print(f"❌ Error: File '{image_path}' not found!")
#         return "File Not Found"

#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),  # ✅ Corrected to 64x64
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
    
#     # ✅ Open Image with PIL (Fixing TypeError)
#     image = Image.open(image_path).convert("RGB")  # Convert to RGB
#     image = transform(image).unsqueeze(0)

#     with torch.no_grad():
#         output = model(image)
#         _, predicted = torch.max(output, 1)           # ✅ Softmax selects the highest probability class

#     # ✅ Load class labels
#     with open("fruits.txt", "r") as f:
#         class_names = [line.strip() for line in f.readlines()]

#     return class_names[predicted.item()]

# # ✅ Run Prediction from Command Line
# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Predict an image")
#     parser.add_argument("image_path", type=str, help="Path to the image")
#     args = parser.parse_args()

#     result = predict_image(args.image_path)
#     print(f"✅ Prediction: {result}")


# #new code with proper image pre-processing 
# import sys
# import os
# import torch
# import torchvision.transforms as transforms
# from PIL import Image

# # ✅ Fix Import Issue
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from models.custom_cnn import FruitVegCNN

# # ✅ Load Model
# model = FruitVegCNN(num_classes=131)
# model.load_state_dict(torch.load('./saved_models/best_model.pth', map_location=torch.device('cpu')))
# model.eval()

# # ✅ Image Processing Function (Fixes Accuracy Issues)
# def process_image(image_path):
#     # ✅ Check if Image Exists
#     if not os.path.exists(image_path):
#         print(f"❌ Error: File '{image_path}' not found!")
#         return None

#     transform = transforms.Compose([
#         transforms.Resize((100, 100)),  # ✅ Changed from 64x64 to 100x100
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # ✅ Fix lighting differences
#         transforms.RandomAdjustSharpness(1.5),  # ✅ Fix image sharpness
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
    
#     # ✅ Open Image with PIL (Fixing TypeError)
#     image = Image.open(image_path).convert("RGB")  # Convert to RGB
#     image = transform(image).unsqueeze(0)  # Add batch dimension

#     return image

# # ✅ Function to Predict Image
# def predict_image(image_path):
#     image_tensor = process_image(image_path)
#     if image_tensor is None:
#         return "File Not Found"

#     with torch.no_grad():
#         output = model(image_tensor)
#         _, predicted = torch.max(output, 1)  # ✅ Softmax selects the highest probability class

#     # ✅ Load class labels
#     with open("fruits.txt", "r") as f:
#         class_names = [line.strip() for line in f.readlines()]

#     return class_names[predicted.item()]

# # ✅ Run Prediction from Command Line
# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Predict an image")
#     parser.add_argument("image_path", type=str, help="Path to the image")
#     args = parser.parse_args()

#     result = predict_image(args.image_path)
#     print(f"✅ Prediction: {result}")



# #new optimzed code 
# import sys
# import os
# import torch
# import torchvision.transforms as transforms
# from PIL import Image

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from models.custom_cnn import FruitVegCNN

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ✅ Load Model
# model = FruitVegCNN(num_classes=131).to(device)
# model.load_state_dict(torch.load('./saved_models/best_model.pth', map_location=device))
# model.eval()

# # ✅ Image Preprocessing Function
# def process_image(image_path):
#     if not os.path.exists(image_path):
#         print(f"❌ Error: File '{image_path}' not found!")
#         return None

#     transform = transforms.Compose([
#         transforms.Resize((100, 100)),  
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])

#     image = Image.open(image_path).convert("RGB")
#     image = transform(image).unsqueeze(0).to(device)  

#     return image

# # ✅ Function to Predict Image
# def predict_image(image_path):
#     image_tensor = process_image(image_path)
#     if image_tensor is None:
#         return "File Not Found"

#     with torch.no_grad():
#         output = model(image_tensor)
#         _, predicted = torch.max(output, 1)  

#     with open("fruits.txt", "r") as f:
#         class_names = [line.strip() for line in f.readlines()]

#     return class_names[predicted.item()]

# # ✅ Run from Command Line
# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Predict an image")
#     parser.add_argument("image_path", type=str, help="Path to the image")
#     args = parser.parse_args()

#     result = predict_image(args.image_path)
#     print(f"✅ Prediction: {result}")


import sys
import os
import torch
import torchvision.transforms as transforms
from PIL import Image

# ✅ Ensure Python Finds the Models Folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.custom_cnn import FruitVegCNN

# ✅ Set Device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔍 Using Device: {device}")

# ✅ Load Model
model_path = os.path.abspath("./saved_models/best_model.pth")
if not os.path.exists(model_path):
    print(f"❌ Error: Model file not found at {model_path}")
    sys.exit(1)

model = FruitVegCNN(num_classes=54).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Model Loaded Successfully!")

# ✅ Load Class Labels
labels_path = os.path.abspath("fruits.txt")
if not os.path.exists(labels_path):
    print(f"❌ Error: fruits.txt not found at {labels_path}")
    sys.exit(1)

with open(labels_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]
print("✅ Class Labels Loaded Successfully!")


# ✅ Image Preprocessing Function
def process_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: File '{image_path}' not found!")
        return None

    print(f"📸 Processing Image: {image_path}")

    transform = transforms.Compose([
        transforms.Resize((100, 100)),  
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  

    print("✅ Image Preprocessing Done!")
    return image


# ✅ Function to Predict Image
def predict_image(image_path):
    image_tensor = process_image(image_path)
    if image_tensor is None:
        return "File Not Found"

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)  

    return class_names[predicted.item()]


# ✅ Run from Command Line
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict an image")
    parser.add_argument("image_path", type=str, help="Path to the image")
    args = parser.parse_args()

    result = predict_image(args.image_path)
    print(f"🎯 Prediction: {result}")
