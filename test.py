import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Path to the model file
model_path = 'new_114.h5'

# Check if the model file exists
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"File not found: {model_path}")

# Load the trained model
model = load_model(model_path)

# Compile the model if necessary (helpful for models loaded without precompiled metrics)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model's input shape to ensure the preprocessing step is correct
print(f"Model input shape: {model.input_shape}")

# Emotion labels (angry, disgust, fear, happy, sad, surprise, neutral) for FER-2013 Dataset
emotion_labels = ['angry', 'sad']

# Define a threshold for prediction confidence (tune this as needed)
confidence_threshold = 0.7  # 70% confidence required for a valid prediction

# Load OpenCV's Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Ensure the camera opened successfully
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Crop the detected face region from the original frame (not the grayscale one)
        face = frame[y:y+h, x:x+w]

        # Resize and preprocess the face for emotion detection
        face_resized = cv2.resize(face, (48, 48))  # Resize to the size expected by the model
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        face_array = np.expand_dims(face_resized, axis=0)  # Add batch dimension
        face_array = np.expand_dims(face_array, axis=-1)   # Add channel dimension
        face_array = face_array / 255.0  # Normalize pixel values to [0, 1]

        # Display the detected face for debugging
        cv2.imshow('Detected Face', face_resized)

        # Make prediction
        predictions = model.predict(face_array)

        # Get the predicted class and confidence score
        predicted_class = np.argmax(predictions[0])
        confidence_score = np.max(predictions[0])

        # Determine the predicted emotion based on confidence
        if confidence_score < confidence_threshold:
            predicted_emotion = "Unknown"
        elif predicted_class < len(emotion_labels):
            predicted_emotion = emotion_labels[predicted_class]
        else:
            predicted_emotion = "Unknown"

        # Display the predicted emotion label on the frame
        cv2.putText(frame, f'Emotion: {predicted_emotion}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the resulting frame
    cv2.imshow('Webcam Feed', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()   