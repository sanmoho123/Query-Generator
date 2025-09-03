import cv2 
import numpy as np
import tensorflow as tf
import dlib
from scipy.spatial import distance
import imutils
from imutils import face_utils
import playsound
import threading

# Load the trained CNN model
model = tf.keras.models.load_model("drowsiness_model.h5")

# Load dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file is available

# Define eye and mouth landmark indices
LEFT_EYE = slice(42, 48)
RIGHT_EYE = slice(36, 42)
MOUTH = slice(60, 68)  # 8 landmark points (60-67)

# Function to compute Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to compute Mouth Aspect Ratio (MAR) (Fixed Indexing)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[6])  # Outer vertical distance
    B = distance.euclidean(mouth[3], mouth[5])  # Inner vertical distance
    C = distance.euclidean(mouth[0], mouth[4])  # Horizontal distance
    return (A + B) / (2.0 * C)

# Function to play alert sound in a separate thread
def play_alarm():
    playsound.playsound("Alert.wav")  # Ensure "Alert.wav" exists

# Constants for thresholds
EAR_THRESHOLD = 0.25  # Eye closing threshold
MAR_THRESHOLD = 0.6  # Yawning threshold (adjusted)
FRAME_COUNT = 20  # Consecutive frames to trigger alert

# Initialize counters
ear_counter = 0
mar_counter = 0

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Compute EAR for both eyes
        left_eye = shape[LEFT_EYE]
        right_eye = shape[RIGHT_EYE]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        # Compute MAR for yawning detection
        mouth = shape[MOUTH]
        mar = mouth_aspect_ratio(mouth)

        # Draw landmarks on eyes and mouth
        cv2.polylines(frame, [left_eye], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(frame, [right_eye], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(frame, [mouth], isClosed=True, color=(0, 0, 255), thickness=2)

        # Check EAR threshold for drowsiness
        if ear < EAR_THRESHOLD:
            ear_counter += 1
            if ear_counter >= FRAME_COUNT:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                threading.Thread(target=play_alarm).start()
        else:
            ear_counter = 0

        # Check MAR threshold for yawning
        if mar > MAR_THRESHOLD:
            mar_counter += 1
            if mar_counter >= FRAME_COUNT:
                cv2.putText(frame, "YAWNING ALERT!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                threading.Thread(target=play_alarm).start()
        else:
            mar_counter = 0

        # Display EAR and MAR values
        cv2.putText(frame, f"EAR: {ear:.2f}", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (400, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Show frame
    cv2.imshow("Drowsiness & Yawning Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
