# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class FruitVegCNN(nn.Module):
#     def __init__(self, num_classes=131):  # Fruits-360 has 131 classes
#         super(FruitVegCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(128 * 12 * 12, 256)  # Adjusted for 100x100 images
#         self.fc2 = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(x.size(0), -1)  # Flatten
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# #new code , last checkpoint
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class FruitVegCNN(nn.Module):
#     def __init__(self, num_classes=131):  # Fruits-360 has 131 classes
#         super(FruitVegCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)

#         # ✅ Corrected for 64x64 images
#         self.fc1 = nn.Linear(128 * 8 * 8, 256)  
#         self.fc2 = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
        
#         x = x.view(x.size(0), -1)  # Flatten
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


# #new code 
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class FruitVegCNN(nn.Module):
#     def __init__(self, num_classes=131):
#         super(FruitVegCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # ✅ Added extra layer
#         self.pool = nn.MaxPool2d(2, 2)

#         # ✅ Corrected for 100x100 images
#         self.fc1 = nn.Linear(256 * 6 * 6, 512)  # ✅ Increased neurons
#         self.fc2 = nn.Linear(512, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = self.pool(F.relu(self.conv4(x)))  # ✅ Extra layer improves feature learning
        
#         x = x.view(x.size(0), -1)  # Flatten
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

#optiimzed code 
import torch
import torch.nn as nn
import torch.nn.functional as F

class FruitVegCNN(nn.Module):
    def __init__(self, num_classes=54):
        super(FruitVegCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # ✅ Batch Normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)

        # ✅ Adaptive layer to handle various input sizes dynamically
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(256 * 6 * 6, 512)
        self.dropout = nn.Dropout(0.5)  # ✅ Dropout to prevent overfitting
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
