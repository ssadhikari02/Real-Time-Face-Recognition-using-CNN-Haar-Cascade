import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json

# Load the trained model
print("Loading model...")
model = load_model("face_recognition_model.h5")
print("Model loaded.")

# Label dictionary (update according to your training labels)
with open("label_dict.json", "r") as f:
    label_dict = json.load(f)

# Reverse mapping: int → name
inv_label_dict = {v: k for k, v in label_dict.items()}

img_size = 100

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
print("Starting webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
else:
    print("Webcam opened successfully.")

cv2.namedWindow("Real-Time Face Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Real-Time Face Recognition", 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        continue

    # Flip the frame horizontally (mirror image)
    frame = cv2.flip(frame, 1)
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (img_size, img_size))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=(0, -1))  # Shape: (1, img_size, img_size, 1)

        prediction = model.predict(face)
        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index]
        label = inv_label_dict.get(class_index, "Unknown")

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{label} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)


    # Show the pop-up window
    cv2.imshow("Real-Time Face Recognition", frame)

    # Exit if 'q' is pressed or window is closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break
    
    if cv2.getWindowProperty("Real-Time Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
        print("Exiting via window close.")
        break
    
cap.release()
cv2.destroyAllWindows()
