# import streamlit as st
# import torch
# import torchvision.transforms as transforms
# import cv2
# from models.custom_cnn import FruitVegCNN

# st.title("Fruit & Vegetable Recognition")

# uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

# if uploaded_file:
#     with open("temp.jpg", "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     model = FruitVegCNN(num_classes=131)
#     model.load_state_dict(torch.load('../saved_models/best_model.pth'))
#     model.eval()

#     transform = transforms.Compose([
#         transforms.Resize((100, 100)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])

#     image = cv2.imread("temp.jpg")
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = transform(image).unsqueeze(0)

#     with torch.no_grad():
#         output = model(image)
#         _, predicted = torch.max(output, 1)

#     st.image("temp.jpg", caption=f"Predicted Class: {predicted.item()}", use_column_width=True)


#new code last checkpoint
# import streamlit as st
# import torch
# import torchvision.transforms as transforms
# import cv2
# import numpy as np
# from PIL import Image
# import sys
# import os

# # ✅ Fix Import Issue
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from models.custom_cnn import FruitVegCNN

# # ✅ Load Model
# model = FruitVegCNN(num_classes=131)
# model.load_state_dict(torch.load("./saved_models/best_model.pth", map_location=torch.device('cpu')))
# model.eval()

# # ✅ Image Processing Function
# def process_image(image):
#     transform = transforms.Compose([
#         transforms.Resize((64, 64)),  # ✅ Changed to 64x64
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#     image = Image.open(image).convert("RGB")  # Convert to RGB
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     return image

# # ✅ Prediction Function
# def predict(image):
#     image_tensor = process_image(image)
#     with torch.no_grad():
#         output = model(image_tensor)
#         _, predicted = torch.max(output, 1)

#     # Load class labels
#     with open("fruits.txt", "r") as f:
#         class_names = [line.strip() for line in f.readlines()]

#     return class_names[predicted.item()]

# # ✅ Streamlit Web App
# st.title("🍎 Fruit & Vegetable Recognition App 🥦")
# st.write("Upload an image to classify whether it's a fruit or vegetable.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# if uploaded_file:
#     st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
#     st.write("Classifying...")

#     result = predict(uploaded_file)
    
#     st.write(f"✅ **Prediction: {result}**")



# #new code 
# import os
# import streamlit as st
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import sys

# # ✅ Fix Import Issue
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from models.custom_cnn import FruitVegCNN

# # ✅ Fix Streamlit's Torch Watcher Issue
# os.environ["PYTORCH_JIT"] = "0"

# # ✅ Load Model
# model = FruitVegCNN(num_classes=54)
# model.load_state_dict(torch.load("./saved_models/best_model.pth", map_location=torch.device('cpu')))
# model.eval()

# # ✅ Image Processing Function
# def process_image(image):
#     transform = transforms.Compose([
#         transforms.Resize((100, 100)),  
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#     image = Image.open(image).convert("RGB")  # Convert to RGB
#     image = transform(image).unsqueeze(0)  # Add batch dimension
#     return image

# # ✅ Prediction Function
# def predict(image):
#     image_tensor = process_image(image)
#     with torch.no_grad():
#         output = model(image_tensor)
#         _, predicted = torch.max(output, 1)

#     # ✅ Load class labels
#     with open("fruits.txt", "r") as f:
#         class_names = [line.strip() for line in f.readlines()]

#     return class_names[predicted.item()]

# # ✅ Streamlit Web App
# st.title("🍎 Fruit & Vegetable Recognition App 🥦")
# st.write("Upload an image to classify whether it's a fruit or vegetable.")

# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# if uploaded_file:
#     st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)  # ✅ FIXED: use_container_width
#     st.write("Classifying...")

#     result = predict(uploaded_file)
    
#     st.write(f"✅ **Prediction: {result}**")




#new trying code for more output 
import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st

# ✅ Fix Import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.custom_cnn import FruitVegCNN

# ✅ Torch JIT Fix
os.environ["PYTORCH_JIT"] = "0"

# ✅ Load Class Labels
with open("fruits.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
num_classes = len(class_names)

# ✅ Load Model
device = torch.device("cpu")
model = FruitVegCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("./saved_models/best_model.pth", map_location=device))
model.eval()

# ✅ Image Transform Function
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image).convert("RGB")
    return transform(image).unsqueeze(0)

# ✅ Prediction Function (Top 3)
def predict(image):
    image_tensor = process_image(image).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        top3_probs, top3_indices = torch.topk(probabilities, 3)
    results = []
    for i in range(3):
        idx = top3_indices[0][i].item()
        prob = top3_probs[0][i].item()
        results.append((class_names[idx], prob))
    return results

# ✅ Sidebar Info
st.sidebar.title("🧠 Model Info")
st.sidebar.write("CNN-based Fruit & Veg Classifier")
st.sidebar.success("Model Trained on 27k+ Images")
st.sidebar.code("Input size: 100x100")
st.sidebar.warning("CPU Mode Active")

# ✅ Main UI
st.title("🍎 Fruit & Vegetable Classifier")
st.markdown("Upload an image below and get the top 3 predictions with confidence.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="🖼️ Uploaded Image", use_column_width=True)
    st.info("Analyzing image...")

    results = predict(uploaded_file)
    top1 = results[0]

    st.success(f"✅ **Top Prediction:** {top1[0]} ({top1[1]*100:.2f}%)")
    
    st.markdown("---")
    st.subheader("🔍 Top 3 Predictions:")

    for label, prob in results:
        st.write(f"🔹 {label}: {prob*100:.2f}%")
        st.progress(min(int(prob * 100), 100))  # Cap at 100%

    # Optional: Fun facts
    fruit_facts = {
    "Apple": "🍏 Apples float because 25% of their volume is air!",
    "Apricot": "🍑 Apricots were first cultivated in China over 4,000 years ago!",
    "Avocado": "🥑 Avocados are technically berries and contain more potassium than bananas!",
    "Avocado ripe": "🥑 Ripe avocados yield slightly to pressure and are perfect for guacamole!",
    "Banana": "🍌 Bananas are berries, but strawberries are not!",
    "Beetroot": "🟥 Beetroot was once used as a natural red dye.",
    "Blueberry": "🫐 Blueberries are one of the only natural foods that are truly blue in color!",
    "Carambula": "⭐ Carambola is also known as star fruit due to its star shape!",
    "Cauliflower": "🥦 Cauliflower comes in purple, green, and orange varieties too!",
    "Chestnut": "🌰 Chestnuts are the only nuts that contain vitamin C.",
    "Cocos": "🥥 The coconut isn't a nut, it's a drupe!",
    "Corn": "🌽 Corn always has an even number of rows on each cob!",
    "Cucumber Ripe": "🥒 Ripe cucumbers can turn yellow and are used for pickling!",
    "Dates": "🌴 Dates have been cultivated for over 6,000 years in the Middle East.",
    "Eggplant": "🍆 Eggplants are related to tomatoes and potatoes.",
    "Fig": "🌰 Figs were one of the first fruits to be cultivated by humans!",
    "Ginger Root": "🧄 Ginger is actually a stem, not a root!",
    "Grape Blue": "🍇 Blue grapes are rich in antioxidants and make bold wines.",
    "Grape Pink": "🍇 Pink grapes are sweet and often used in juices.",
    "Grape White": "🍇 White grapes are typically green and used for white wines.",
    "Grape White 2": "🍇 Grape White 2 is a variety commonly used in raisins!",
    "Guava": "🍈 Guava has 4 times more vitamin C than an orange!",
    "Hazelnut": "🌰 Hazelnuts are often used to make Nutella.",
    "Huckleberry": "🫐 Huckleberries are wild berries native to North America.",
    "Kiwi": "🥝 Kiwi was once known as 'Chinese Gooseberry'!",
    "Lemon": "🍋 Lemons contain more sugar than strawberries!",
    "Limes": "🍈 Limes were used by sailors to prevent scurvy.",
    "Lychee": "🍒 Lychee is known as the 'fruit of romance' in China.",
    "Mango": "🥭 Mango is the national fruit of India and Pakistan!",
    "Mulberry": "🫐 Silkworms feed exclusively on mulberry leaves!",
    "Onion Red Peeled": "🧅 Red onions are sweeter and milder than yellow ones.",
    "Onion White": "🧄 White onions have a sharp, tangy flavor and are used in salsas.",
    "Orange": "🍊 Oranges are the most cultivated fruit tree in the world.",
    "Papaya": "🥭 Papaya contains papain, an enzyme that helps digestion.",
    "Peach": "🍑 Peaches originated in China and symbolize immortality.",
    "Pear": "🍐 Pears ripen from the inside out.",
    "Pear Forelle": "🍐 Forelle pears have speckled skin and sweet, spicy flavor.",
    "Pepper Green": "🫑 Green peppers are just unripe red or yellow ones!",
    "Pepper Orange": "🫑 Orange peppers are the sweetest of all bell peppers.",
    "Pepper Red": "🫑 Red bell peppers have the highest vitamin C content.",
    "Pepper Yellow": "🫑 Yellow peppers have a tangy and sweet flavor.",
    "Pineapple": "🍍 Pineapples take 2 years to grow and don’t ripen after harvest!",
    "Pineapple Mini": "🍍 Mini pineapples are sweeter and entirely edible!",
    "Pitahaya Red": "🐉 Also known as Dragon Fruit, it's rich in fiber and magnesium.",
    "Plum": "🍑 Plums are one of the first fruits domesticated by humans.",
    "Pomegranate": "🍎 Pomegranates can contain up to 1,400 seeds!",
    "Potato Red": "🥔 Red potatoes are great for roasting due to their waxy texture.",
    "Potato White": "🥔 White potatoes are all-purpose and mild-flavored.",
    "Raspberry": "🍓 Raspberries have a hollow core unlike blackberries.",
    "Strawberry": "🍓 Strawberries are the only fruit with seeds on the outside!",
    "Tomato 2": "🍅 Tomatoes were once considered poisonous!",
    "Tomato Cherry Red": "🍅 Cherry tomatoes are bite-sized and sweeter than regular tomatoes.",
    "Walnut": "🌰 Walnuts resemble the human brain and are great for brain health.",
    "Watermelon": "🍉 Watermelons are 92% water!"
    }


    if top1[0] in fruit_facts:
        st.info(f"💡 Fun Fact: {fruit_facts[top1[0]]}")
