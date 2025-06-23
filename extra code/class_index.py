#use this file to generate the classes with index and compare it with fruits.txt

import os

dataset_path = ".\dataset_fruit_360/train"
categories = sorted(os.listdir(dataset_path))

for idx, category in enumerate(categories):
    print(f"{idx}: {category}")
