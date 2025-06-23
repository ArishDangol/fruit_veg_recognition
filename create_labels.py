import os

dataset_path = "dataset_fruit_360/train"  # Path to the training folder
categories = sorted(os.listdir(dataset_path))  # Get all class names

# Save class labels in a file
with open("fruits.txt", "w") as f:
    for label in categories:
        f.write(label + "\n")

print("fruits.txt created successfully!")
