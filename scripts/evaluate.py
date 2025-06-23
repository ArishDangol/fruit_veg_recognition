# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from models.custom_cnn import FruitVegCNN
# import sys
# import os

# # ✅ Fix Import Issue
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # ✅ Load the Trained Model
# model = FruitVegCNN(num_classes=131)
# model.load_state_dict(torch.load("./saved_models/best_model.pth", map_location=torch.device('cpu')))
# model.eval()

# # ✅ Data Preprocessing (Updated to 64x64)
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),  
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# # ✅ Load Test Dataset
# testset = torchvision.datasets.ImageFolder(root='./dataset_fruit_360/test', transform=transform)
# testloader = DataLoader(testset, batch_size=32, shuffle=False)

# # ✅ Evaluate Model Accuracy
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels in testloader:
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)           # ✅ Softmax selects the highest probability class
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# accuracy = 100 * correct / total
# print(f'✅ Model Evaluation Completed! Accuracy: {accuracy:.2f}%')


# #optimzed code 
# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from models.custom_cnn import FruitVegCNN
# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ✅ Load Trained Model
# model = FruitVegCNN(num_classes=131).to(device)
# model.load_state_dict(torch.load("./saved_models/best_model.pth", map_location=device))
# model.eval()

# # ✅ Preprocessing (Consistent with Training)
# transform = transforms.Compose([
#     transforms.Resize((100, 100)),  
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# # ✅ Load Test Dataset
# testset = torchvision.datasets.ImageFolder(root='./dataset_fruit_360/test', transform=transform)
# testloader = DataLoader(testset, batch_size=64, shuffle=False)

# # ✅ Evaluate Model
# correct = 0
# total = 0
# with torch.no_grad():
#     for images, labels in testloader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)  
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# accuracy = 100 * correct / total
# print(f'✅ Model Evaluation Completed! Accuracy: {accuracy:.2f}%')


# #✅ CPU-Optimized Evaluation Code
# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from models.custom_cnn import FruitVegCNN
# import sys
# import os
# from tqdm import tqdm  # ✅ Progress bar

# # ✅ Fix Import Issues
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # ✅ Force CPU Mode (No GPU)
# device = torch.device("cpu")  # 🚀 Now using ONLY CPU

# # ✅ Load Model
# model = FruitVegCNN(num_classes=131).to(device)
# model.load_state_dict(torch.load("./saved_models/best_model.pth", map_location=device))
# model.eval()

# # ✅ Data Preprocessing (Matches Training)
# transform = transforms.Compose([
#     transforms.Resize((100, 100)),  
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# # ✅ Load Test Dataset
# testset = torchvision.datasets.ImageFolder(root='./dataset_fruit_360/test', transform=transform)

# # ✅ Reduce Batch Size for CPU (Speeds Up Processing)
# testloader = DataLoader(testset, batch_size=32, shuffle=False)  

# # ✅ Evaluate Model with Progress Bar
# correct = 0
# total = 0

# print("🔎 Evaluating Model (CPU Mode)...")
# with torch.no_grad():
#     for images, labels in tqdm(testloader, desc="⏳ Processing Batches"):
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)  
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# # ✅ Display Final Accuracy
# accuracy = 100 * correct / total
# print(f'🎯 Model Evaluation Completed! Accuracy: {accuracy:.2f}% ✅')





#✅ CPU-Optimized Evaluation Code
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.custom_cnn import FruitVegCNN
import sys
import os
from tqdm import tqdm  # ✅ Progress bar

# ✅ Fix Import Issues
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ✅ Force CPU Mode (No GPU)
device = torch.device("cpu")  # 🚀 Now using ONLY CPU

# ✅ Load Model
model = FruitVegCNN(num_classes=54).to(device)
model.load_state_dict(torch.load("./saved_models/best_model.pth", map_location=device))
model.eval()

# ✅ Data Preprocessing (Matches Training)
transform = transforms.Compose([
    transforms.Resize((100, 100)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ✅ Load Test Dataset
testset = torchvision.datasets.ImageFolder(root='./dataset_fruit_360/test', transform=transform)

# ✅ Reduce Batch Size for CPU (Speeds Up Processing)
testloader = DataLoader(testset, batch_size=32, shuffle=False)  

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ✅ After your prediction loop:
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in tqdm(testloader, desc="⏳ Processing Batches"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# ✅ Classification Report
print("\n📄 Classification Report:")
print(classification_report(y_true, y_pred))

# ✅ Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="Blues", xticklabels=testset.classes, yticklabels=testset.classes, annot=False, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("🍓 Confusion Matrix")
plt.tight_layout()
plt.show()
