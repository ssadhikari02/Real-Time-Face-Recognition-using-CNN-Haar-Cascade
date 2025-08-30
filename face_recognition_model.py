import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import json
# Parameters
img_size = 100
data_dir = "dataset"

def load_data():
    images = []
    labels = []
    label_dict = {}
    label_count = 0

    for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir):
            continue
        if person not in label_dict:
            label_dict[person] = label_count
            label_count += 1
        label = label_dict[person]
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(label)

    images = np.array(images).reshape(-1, img_size, img_size, 1) / 255.0
    labels = to_categorical(np.array(labels))
    return images, labels, label_dict

# Load data
X, y, label_dict = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_dict), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save label dictionary
with open("label_dict.json", "w") as f:
    json.dump(label_dict, f)

# Save model
model.save("face_recognition_model.h5")