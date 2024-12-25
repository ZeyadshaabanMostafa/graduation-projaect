import cv2
import torch
import os
import json
import csv
import torch.nn.functional as F
import torchvision.transforms as transforms
from tkinter import Tk, filedialog, Button, Label, Canvas
from torchvision import models
from PIL import Image, ImageTk
import face_recognition
import numpy as np

# Load the pre-trained AlexNet model
alexnet = models.alexnet(pretrained=True)
alexnet.eval()  # Set the model to evaluation mode

# Preprocessing function for image (to match ImageNet input format)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet class labels
with open("imagenet_class_index.json") as f:
    class_idx = json.load(f)

# Function to load an image from file
def load_image():
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    return file_path if file_path else None

# Function to detect faces in an image using OpenCV
def detect_faces(image_path):
    print(f"Loading image from: {image_path}")  # Debugging line
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")  # Debugging line
        return [], None  # Return empty if image can't be loaded
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces, img

# Function to process image and make predictions
def recognize_face(image_path, known_face_encodings, criminal_face_encodings, canvas, label, name_label):
    faces, img = detect_faces(image_path)
    
    if len(faces) == 0:
        label.config(text="No faces detected.")
        name_label.config(text="")
        return

    name = "Unknown Person"

    # Convert the image to RGB for face_recognition compatibility
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect face encodings using face_recognition
    face_locations = face_recognition.face_locations(img_rgb)  # Get face locations (bounding boxes)
    face_encodings = face_recognition.face_encodings(img_rgb, face_locations)  # Get encodings for detected faces

    if not face_encodings:
        label.config(text="No faces recognized.")
        name_label.config(text="")
        return

    # Iterate over detected faces
    for (x, y, w, h), face_encoding in zip(faces, face_encodings):
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Check if the face is a known criminal
        matches = face_recognition.compare_faces(criminal_face_encodings, face_encoding)
        if True in matches:
            name = "Criminal"
            cv2.putText(img, "Criminal", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            # Check if the face is a known non-criminal
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in matches:
                name = "Known Person"
                cv2.putText(img, "Known Person", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(img, "Unknown Person", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image = img_tk

    label.config(text="Face recognition completed!")
    name_label.config(text=name)

# Function to load encodings from CSV
def load_encodings_from_csv(file_path):
    encodings = []
    names = []
    try:
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                name = row[0]
                encoding = np.array([float(val) for val in row[1].split(',')])  # Convert string back to float array
                encodings.append(encoding)
                names.append(name)
    except Exception as e:
        print(f"Error loading encodings from CSV: {e}")
    return encodings, names

# Main GUI application
def main():
    root = Tk()
    root.title("Face Recognition and Classification")

    # Create a canvas to display the image
    canvas = Canvas(root, width=800, height=600)
    canvas.pack()

    # Create a label for messages
    label = Label(root, text="Select an image to begin", font=("Arial", 14))
    label.pack(pady=10)

    # Create a label to display the name of the recognized person
    name_label = Label(root, text="", font=("Arial", 14))
    name_label.pack(pady=10)

    # Load known faces (non-criminals) and criminal faces from CSV files
    known_face_encodings, known_face_names = load_encodings_from_csv('Non-Criminal_faces.csv')
    criminal_face_encodings, criminal_face_names = load_encodings_from_csv('Criminal_faces.csv')

    # Debugging output to verify loading
    print("Known Face Encodings Loaded:")
    print(known_face_encodings)
    print("Criminal Face Encodings Loaded:")
    print(criminal_face_encodings)

    if not known_face_encodings and not criminal_face_encodings:
        label.config(text="No faces loaded!")
        return

    # Function to handle the upload button action
    def on_upload():
        image_path = load_image()
        if image_path:
            recognize_face(image_path, known_face_encodings, criminal_face_encodings, canvas, label, name_label)

    # Function to handle re-upload and re-recognition
    def on_reupload():
        canvas.delete("all")  # Clear the previous image
        label.config(text="Select an image to begin")
        name_label.config(text="")
        on_upload()  # Re-upload and recognize

    upload_button = Button(root, text="Upload Image", font=("Arial", 14), command=on_upload)
    upload_button.pack(pady=10)

    reupload_button = Button(root, text="Re-upload Image", font=("Arial", 14), command=on_reupload)
    reupload_button.pack(pady=10)

    root.mainloop()

# Run the application
if __name__ == "__main__":
    main()