import os
import cv2

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier('Face_Recognition/Data_files/haarcascade_frontalface_default.xml')

# Load the trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
xml_file_path = 'Face_Recognition/Face_jpg/trained_recognizer.xml'
recognizer.read(xml_file_path)

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop the face region
        face_roi = gray[y:y + h, x:x + w]

        # Recognize the face
        label, confidence = recognizer.predict(face_roi)

        # Get the base name (remove extension) from the XML file
        label_name = os.path.splitext(os.path.basename(xml_file_path))[0]

        # Display the recognized face and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label_name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()
