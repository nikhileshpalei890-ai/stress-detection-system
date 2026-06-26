import cv2
import numpy as np
import streamlit as st
from keras.models import load_model

# Set up page config
st.set_page_config(page_title="Emotion Detection System", layout="centered")
st.title("Real-Time Emotion Detection System")
st.write("This application uses a Deep Learning CNN model to monitor and track emotional states live.")

# 1. Load model safely (Uses relative path for cloud deployment)
@st.cache_resource
def load_emotion_model():
    # Make sure 'Emotion_detection_model_new.h5' is uploaded to the root folder of your GitHub repository
    return load_model("Emotion_detection_model_new.h5")

try:
    model = load_emotion_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model. Make sure the model file is in the same directory. Details: {e}")
    st.stop()

# 2. Emotion labels matching your training code
emotion_labels = {
    0: "Angry",
    1: "Happy",
    2: "Neutral",
    3: "Sad",
    4: "Surprised"
}

# 3. Load face detector using standard OpenCV cascade paths
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 4. User interface controls
run_app = st.checkbox("Turn on Webcam Feed")
FRAME_WINDOW = st.image([]) # Placeholder container for the video frames

if run_app:
    # Open local webcam channel
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not access or open webcam device.")
    
    while run_app:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera frame not detected.")
            break
            
        # Optional: Flip image for natural mirroring effect
        frame = cv2.flip(frame, 1)
        
        # Preprocessing pipeline matching your exact code
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            # Crop, resize, normalize, and reshape
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = face.reshape(1, 48, 48, 1)
            
            # Run Inference
            prediction = model.predict(face, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]
            
            # Draw green bounding box and dynamic overlay text on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame, 
                emotion, 
                (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (0, 255, 0), 
                2
            )
            
        # Convert frame color format from BGR (OpenCV standard) to RGB (Web/Streamlit standard)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Stream the modified frame directly to the web UI placeholder
        FRAME_WINDOW.image(frame_rgb)
        
    # Clean up hardware resources cleanly on toggle-off
    cap.release()
else:
    st.info("Check the box above to activate your webcam and start scanning expressions.")
