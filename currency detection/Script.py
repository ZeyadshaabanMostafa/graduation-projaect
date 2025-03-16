import os
import cv2
import numpy as np
from tkinter import filedialog, Label, Button, Entry, Tk
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

# ✅ Load model and dataset structure
device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_path = "E:/currency_data"
classes = sorted(os.listdir(dataset_path))

# ✅ Define model architecture
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("currency_feature_detector.pth", map_location=device))
model.to(device)
model.eval()

# ✅ Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
])

# ✅ Function to extract features and compare with stored currency features
def search_for_currency_features(image_path, currency_number):
    currency_folder = os.path.join(dataset_path, str(currency_number))

    if not os.path.exists(currency_folder):
        return "Error: Currency folder not found!"

    # Load the image to search for features
    input_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Iterate through all feature images in the corresponding currency folder
    for feature_img_name in os.listdir(currency_folder):
        feature_img_path = os.path.join(currency_folder, feature_img_name)

        # Load feature image
        feature_img = cv2.imread(feature_img_path, cv2.IMREAD_GRAYSCALE)

        # Perform template matching to find the feature in the input image
        result = cv2.matchTemplate(input_img, feature_img, cv2.TM_CCOEFF_NORMED)

        # Get the matching result
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # If the match is above a certain threshold, it is a valid feature
        threshold = 0.8  # Adjust this threshold value based on experimentation
        if max_val >= threshold:
            return f"✅ Feature matched! Currency seems to be REAL ({feature_img_name})"

    return "❌ No matching features found! Currency might be FAKE."


# ✅ GUI Functions
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        classify_button.config(state="normal")
        global selected_image
        selected_image = file_path

def classify_image():
    if selected_image:
        currency_number = currency_input.get()
        if not currency_number.isdigit():
            result_label.config(text="Error: Enter a valid currency number!")
            return
        result = search_for_currency_features(selected_image, currency_number)
        result_label.config(text=f"Result: {result}")

# ✅ Create GUI
root = Tk()
root.title("Currency Feature Search")

Label(root, text="Currency Feature Recognition System", font=("Arial", 16)).pack()
panel = Label(root)
panel.pack()
Button(root, text="Upload Image", command=upload_image).pack()

Label(root, text="Enter Currency Number:", font=("Arial", 12)).pack()
currency_input = Entry(root)
currency_input.pack()

classify_button = Button(root, text="Check Real/Fake", command=classify_image, state="disabled")
classify_button.pack()

result_label = Label(root, text="", font=("Arial", 14))
result_label.pack()

root.mainloop()
