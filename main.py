import cv2
import numpy as np
import os

# --- SETUP ---
# We use Haar Cascade because it's great for cropping faces for training
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create the Face Recognizer (LBPH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Variables
training_data = []
labels = []
is_trained = False
capture_count = 0
MAX_SAMPLES = 50  # How many photos to take for registration

print("--- SECURITY SYSTEM STARTED ---")
print("1. Press 'r' to REGISTER your face (Look at camera, move head slightly).")
print("2. Press 'q' to QUIT.")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Drawing the UI text
    cv2.putText(frame, "Press 'r' to Register Authorized Face", (10, 30), 
                cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 0), 1)

    for (x, y, w, h) in faces:
        # Get the face region of interest (ROI)
        face_roi = gray[y:y+h, x:x+w]

        # --- MODE 1: REGISTRATION (Collecting Data) ---
        if capture_count > 0 and capture_count <= MAX_SAMPLES:
            training_data.append(face_roi)
            labels.append(1)  # Label '1' is the Authorized User
            capture_count += 1
            
            # Visual feedback
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.putText(frame, f"Scanning: {capture_count}/{MAX_SAMPLES}", (x, y-10), 
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)

            if capture_count == MAX_SAMPLES:
                print("--- TRAINING MODEL... ---")
                recognizer.train(training_data, np.array(labels))
                is_trained = True
                print("--- REGISTRATION COMPLETE. SYSTEM ARMED. ---")

        # --- MODE 2: RECOGNITION (Security Active) ---
        elif is_trained:
            # Predict who this face is
            label, confidence = recognizer.predict(face_roi)
            
            # Confidence Logic: Lower is better (0 = perfect match)
            # Usually < 50 is a very good match, > 80 is likely unknown
            if confidence < 70:  
                name = "AUTHORIZED"
                color = (0, 255, 0) # Green
            else:
                name = "UNAUTHORIZED"
                color = (0, 0, 255) # Red

            # Draw the box and name
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)
            # Optional: Show confidence score (for debugging)
            # cv2.putText(frame, str(round(confidence)), (x, y+h+20), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

        # --- MODE 3: IDLE (Waiting) ---
        else:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 200, 200), 2)
            cv2.putText(frame, "Unknown (System Not Armed)", (x, y-10), 
                        cv2.FONT_HERSHEY_PLAIN, 1, (200, 200, 200), 1)

    cv2.imshow('Security Feed', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and not is_trained:
        capture_count = 1  # Start capturing
        training_data = [] # Reset data
        labels = []

cap.release()
cv2.destroyAllWindows()