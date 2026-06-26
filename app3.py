import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode

st.set_page_config(page_title="Real-Time Emotion Detection", layout="centered")
st.title("Real-Time Emotion Detection System")
st.write("Live video analysis for tracking student stress and emotional states.")

# 1. Load model safely
@st.cache_resource
def load_emotion_model():
    return tf.keras.models.load_model("Emotion_detection_model_new.h5")

try:
    model = load_emotion_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

emotion_labels = {0: "Angry", 1: "Happy", 2: "Neutral", 3: "Sad", 4: "Surprised"}
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 2. Modern Callback Function for Video Frames
def video_frame_callback(frame):
    # Convert WebRTC incoming frame directly to an RGB array
    img = frame.to_ndarray(format="bgr24")
    
    # Preprocessing pipeline
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = face.reshape(1, 48, 48, 1)
        
        # Run Inference
        prediction = model.predict(face, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        # Draw green bounding box and text overlay
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(
            img, 
            f"{emotion} ({confidence:.1f}%)", 
            (x, y-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (0, 255, 0), 
            2
        )
        
    # Re-package the array back into a secure video frame object
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# 3. Streamlit WebRTC Streamer
webrtc_streamer(
    key="emotion-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
