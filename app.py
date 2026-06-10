from flask import Flask, render_template, Response, request
import cv2
import numpy as np
import os

app = Flask(__name__)

# --- CONFIGURATION ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Global variables
camera = cv2.VideoCapture(0)
training_mode = False
capture_count = 0
MAX_SAMPLES = 50
training_data = []
labels = []
model_exists = False

# 1. LOAD PREVIOUS DATA (The Memory)
if os.path.exists('trainer.yml'):
    recognizer.read('trainer.yml')
    model_exists = True
    print("--- MEMORY LOADED: SYSTEM ARMED ---")

def generate_frames():
    global training_mode, capture_count, training_data, labels, model_exists

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]

            # --- LOGIC A: TRAINING (Registration) ---
            if training_mode:
                if capture_count < MAX_SAMPLES:
                    training_data.append(face_roi)
                    labels.append(1) # Label 1 = Authorized
                    capture_count += 1
                    cv2.putText(frame, f"Scanning: {capture_count}%", (x, y-10), 
                                cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                else:
                    # Training finished
                    recognizer.train(training_data, np.array(labels))
                    recognizer.save('trainer.yml') # <--- SAVES DATA PERMANENTLY
                    model_exists = True
                    training_mode = False
                    capture_count = 0
                    print("--- SAVED TO TRAINER.YML ---")

            # --- LOGIC B: RECOGNITION (Security) ---
            elif model_exists:
                label, confidence = recognizer.predict(face_roi)
                
                # Confidence: 0 = Perfect Match, >100 = No Match
                if confidence < 70:
                    name = "AUTHORIZED"
                    color = (0, 255, 0) # Green
                else:
                    name = "UNAUTHORIZED"
                    color = (0, 0, 255) # Red
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

            # --- LOGIC C: IDLE ---
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 2)
                cv2.putText(frame, "No Data - Please Register", (x, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (100, 100, 100), 1)

        # Encode frame for web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['POST'])
def register():
    global training_mode, training_data, labels
    training_mode = True
    training_data = [] # Clear old temp data
    labels = []
    return "Registration Started"

if __name__ == "__main__":
    app.run(debug=True)