# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from models.custom_cnn import FruitVegCNN
# import torch.optim as optim
# import torch.nn as nn
# from tqdm import tqdm
# import sys
# import os

# # ✅ Fix Import Issue (Ensure Python Finds "models/" Folder)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # ✅ Load Custom CNN Model
# from models.custom_cnn import FruitVegCNN

# # ✅ Data Preprocessing (Resizing images to 64x64 for faster training)
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),  
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# # ✅ Load Full Dataset (Best Accuracy)
# trainset = torchvision.datasets.ImageFolder(root='./dataset_fruit_360/train', transform=transform)

# # ✅ Alternative: Use a 30% Subset (Faster Training, Slight Accuracy Drop)
# # subset_size = int(len(trainset) * 0.3)  # Use only 30% of data
# # trainset, _ = torch.utils.data.random_split(trainset, [subset_size, len(trainset) - subset_size])

# trainloader = DataLoader(trainset, batch_size=128, shuffle=True)  # ✅ Increased batch size to 128

# # ✅ Model Setup (No GPU Code)
# model = FruitVegCNN(num_classes=131)

# # ✅ Loss and Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)       

# # ✅ Training Loop (Reduced Epochs from 10 → 5)
# epochs = 5  
# for epoch in range(epochs):
#     running_loss = 0.0
#     for images, labels in tqdm(trainloader):
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
    
#     print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(trainloader)}")

# # ✅ Save Trained Model
# torch.save(model.state_dict(), './saved_models/best_model.pth')

# print("✅ Training Completed! Model Saved Successfully.")



# #new code , 64 to 100 epochs to 10 
# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from models.custom_cnn import FruitVegCNN
# import torch.optim as optim
# import torch.nn as nn
# from tqdm import tqdm
# import sys
# import os

# # ✅ Fix Import Issue
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # ✅ Load Custom CNN Model
# from models.custom_cnn import FruitVegCNN

# # ✅ Data Preprocessing (64 → 100 for better accuracy)
# transform = transforms.Compose([
#     transforms.Resize((100, 100)),  
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),  # ✅ More variety in rotations
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # ✅ Simulate real-world lighting
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# # ✅ Load Full Dataset
# trainset = torchvision.datasets.ImageFolder(root='./dataset_fruit_360/train', transform=transform)

# trainloader = DataLoader(trainset, batch_size=64, shuffle=True)  # ✅ Lower batch size (More stability)

# # ✅ Model Setup
# model = FruitVegCNN(num_classes=131)

# # ✅ Loss and Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0005)  # ✅ Lower learning rate for better stability

# # ✅ Training Loop (Increased Epochs from 5 → 10)
# epochs = 10  
# for epoch in range(epochs):
#     running_loss = 0.0
#     for images, labels in tqdm(trainloader):
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
    
#     print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(trainloader)}")

# # ✅ Save Trained Model
# torch.save(model.state_dict(), './saved_models/best_model.pth')

# print("✅ Training Completed! Model Saved Successfully.")


# #optimzed code 
# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from models.custom_cnn import FruitVegCNN
# import torch.optim as optim
# import torch.nn as nn
# from tqdm import tqdm
# import sys
# import os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# # ✅ Data Augmentation & Preprocessing
# transform = transforms.Compose([
#     transforms.Resize((100, 100)),  
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(20),  
#     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  
#     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # ✅ Better augmentation
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# # ✅ Load Dataset
# trainset = torchvision.datasets.ImageFolder(root='./dataset_fruit_360/train', transform=transform)
# trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

# # ✅ Initialize Model, Loss, and Optimizer
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = FruitVegCNN(num_classes=131).to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # ✅ Learning rate decay

# # ✅ Training Loop
# epochs = 10  
# for epoch in range(epochs):
#     running_loss = 0.0
#     model.train()
#     for images, labels in tqdm(trainloader):
#         images, labels = images.to(device), labels.to(device)
        
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
    
#     scheduler.step()  # ✅ Update learning rate
#     print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(trainloader)}")

# # ✅ Save Model
# torch.save(model.state_dict(), './saved_models/best_model.pth')
# print("✅ Training Completed! Model Saved.")



import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.custom_cnn import FruitVegCNN
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ✅ Image Preprocessing & Augmentation
transform = transforms.Compose([
    transforms.Resize((100, 100)),  # ✅ Increased image size
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ✅ Load Full Dataset (No 30% Restriction)
trainset = torchvision.datasets.ImageFolder(root='./dataset_fruit_360/train', transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

# ✅ Setup Model
device = torch.device("cpu")  # Use CUDA if available
model = FruitVegCNN(num_classes=len(trainset.classes)).to(device)  # Dynamically count classes

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Slightly lowered LR for full dataset
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.6)

# ✅ Training Loop
epochs = 10  # Increased epochs to allow learning from full data
for epoch in range(epochs):
    running_loss = 0.0
    model.train()
    
    for images, labels in tqdm(trainloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    scheduler.step()
    print(f"📊 Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(trainloader):.4f}")

# ✅ Save Model
torch.save(model.state_dict(), './saved_models/best_model.pth')
print("✅ Training Completed! Model Saved.")
