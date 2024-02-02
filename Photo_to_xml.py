import cv2
import os
import numpy as np

def detect_faces(img, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    return faces, gray

def assign_labels(label_dict, root, current_label):
    label = label_dict.setdefault(root, current_label)
    current_label += 1
    return label, current_label

# Load pre-trained face detector (Haar cascade)
face_cascade = cv2.CascadeClassifier('Face_Recognition/Data_files/haarcascade_frontalface_default.xml')

# Directory containing face images
data_dir = 'Face_Recognition/Face_jpg'

# Lists to store face samples and labels
face_samples = []
labels = []

# Dictionary to assign labels to each person (assuming one folder per person)
label_dict = {}
current_label = 0

# Loop through the directory structure to process face images
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.jpg'):
            img_path = os.path.join(root, file)
            
            # Assign label based on folder structure
            label, current_label = assign_labels(label_dict, root, current_label)
            print(f"File: {file}, Label: {label}")

            # Read the image and convert it to grayscale
            img = cv2.imread(img_path)

            # Detect faces in the image
            faces, gray = detect_faces(img, face_cascade)
            print(f"Number of faces detected: {len(faces)}")

            # Extract face samples and corresponding labels
            for (x, y, w, h) in faces:
                face_samples.append(gray[y:y + h, x:x + w])
                labels.append(label)

# Convert the labels to a NumPy array
labels = np.array(labels)

# Create the LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Check if there are enough faces for training
if len(face_samples) < 2:
    print("Error: Not enough faces detected for training.")
else:
    # Train the recognizer
    recognizer.train(face_samples, labels)
    
    # Save the trained recognizer to an XML file
    recognizer.save('Face_Recognition/Face_jpg/trained_recognizer.xml')
