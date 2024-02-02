import cv2

def detect_and_save_faces(frame, face_cascade, photo_count):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=10)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        cv2.imwrite(f'Face_Recognition/Face_jpg/face_{photo_count}.jpg', face_roi)
        photo_count += 1

    return photo_count

def main():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('Face_Recognition/Data_files/haarcascade_frontalface_default.xml')
    photo_count = 0

    while photo_count < 6:
        ret, frame = cap.read()

        photo_count = detect_and_save_faces(frame, face_cascade, photo_count)

        cv2.imshow('Capturing Face', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
