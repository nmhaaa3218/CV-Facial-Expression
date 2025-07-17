import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import os
import cv2
from ultralytics import YOLO
import tempfile

# ---- User Instructions ----
st.title('Facial Expression Recognition (Ensemble)')
st.write('Upload a face image or use webcam to identify emotions using an ensemble of deep learning models. The app shows both the ensembled and individual model predictions.')

# ---- Label Mapping ----
LABELS = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

# ---- Image Preprocessing ----
def preprocess_image(img):
    # Convert to grayscale, resize to 48x48, normalize as in notebook
    img = ImageOps.grayscale(img)
    img = img.resize((48, 48))
    img = np.array(img).astype(np.uint8)
    img = np.expand_dims(img, axis=2)  # (48, 48, 1)
    transform = T.Compose([
        T.ToPILImage(),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # (1, 1, 48, 48)
    return img

# ---- YOLO Face Detection ----
@st.cache_resource
def load_yolo_model():
    # Load YOLO model for face detection
    model = YOLO("yolov12n-face.pt")
    return model

def detect_faces(image, yolo_model):
    """Detect faces in image using YOLO"""
    results = yolo_model(image)
    faces = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                # Filter for high confidence detections
                if confidence > 0.5:
                    faces.append((int(x1), int(y1), int(x2), int(y2), confidence))
    
    return faces

def crop_face(image, bbox):
    """Crop face from image using bounding box with square padding to include hair"""
    x1, y1, x2, y2, _ = bbox
    
    # Calculate current face dimensions
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Add considerable padding (50% on each side)
    padding_factor = 0.5
    padding_x = int(face_width * padding_factor)
    padding_y = int(face_height * padding_factor)
    
    # Expand bounding box with padding
    padded_x1 = max(0, x1 - padding_x)
    padded_y1 = max(0, y1 - padding_y)
    padded_x2 = min(image.width, x2 + padding_x)
    padded_y2 = min(image.height, y2 + padding_y)
    
    # Calculate dimensions of padded region
    padded_width = padded_x2 - padded_x1
    padded_height = padded_y2 - padded_y1
    
    # Make it square by using the larger dimension
    square_size = max(padded_width, padded_height)
    
    # Center the square crop
    center_x = (padded_x1 + padded_x2) // 2
    center_y = (padded_y1 + padded_y2) // 2
    
    # Calculate square crop coordinates
    half_size = square_size // 2
    square_x1 = max(0, center_x - half_size)
    square_y1 = max(0, center_y - half_size)
    square_x2 = min(image.width, center_x + half_size)
    square_y2 = min(image.height, center_y + half_size)
    
    # Crop the square region
    face = image.crop((square_x1, square_y1, square_x2, square_y2))
    return face

# ---- Model Loading ----
@st.cache_resource
def load_models():
    # Import model classes from notebook code (must be available in this file or imported)
    from model import EfficientNetB0, to_device
    device = torch.device('cpu')
    models = []
    model_names = []
    weights = []
    model_eff_b0 = to_device(EfficientNetB0(7,1),device)
    model_eff_b0_2 = to_device(EfficientNetB0(7,1),device)
    model_eff_b0.load_state_dict(torch.load('model_efficientnetb0_v2_tuned_2.pth', map_location=device))
    model_eff_b0_2.load_state_dict(torch.load('model_efficientnetb0_v5_tuned.pth', map_location=device))
    model_eff_b0.eval()
    model_eff_b0_2.eval()
    models.append(model_eff_b0)
    models.append(model_eff_b0_2)
    model_names.append('EfficientNetB0_v2_tuned_2')
    model_names.append('EfficientNetB0_v5_tuned')
    weights.append(0.45)
    weights.append(0.55)
    return models, model_names, weights

# ---- Weighted Average Ensemble ----
def weighted_average_probs(models, img_tensor, weights):
    probs_list = []
    for model in models:
        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)
            probs_list.append(probs.cpu().numpy())
    probs_array = np.stack(probs_list, axis=0)  # (num_models, 1, num_classes)
    weights = np.array(weights).reshape(-1, 1, 1)
    weighted_probs = (probs_array * weights).sum(axis=0)  # (1, num_classes)
    return weighted_probs[0], probs_list

def process_and_predict(face_image, models, model_names, weights):
    """Process face image and make predictions"""
    img_tensor = preprocess_image(face_image)
    ensembled_probs, all_model_probs = weighted_average_probs(models, img_tensor, weights)
    ensembled_pred = int(np.argmax(ensembled_probs))
    return ensembled_probs, all_model_probs, ensembled_pred

def draw_face_boxes(image, faces, predictions=None):
    """Draw bounding boxes around detected faces with emotion labels"""
    draw = ImageDraw.Draw(image)
    
    for i, face in enumerate(faces):
        x1, y1, x2, y2, confidence = face
        
        # Calculate the square crop that actually feeds into the model
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        face_width = x2 - x1
        face_height = y2 - y1
        size = max(face_width, face_height)
        half_size = size // 2
        
        # Calculate square crop bounds (same as crop_face function)
        square_x1 = max(0, center_x - half_size)
        square_y1 = max(0, center_y - half_size)
        square_x2 = min(image.width, center_x + half_size)
        square_y2 = min(image.height, center_y + half_size)
        
        # Draw the actual square crop that feeds into the model
        draw.rectangle([square_x1, square_y1, square_x2, square_y2], outline="red", width=2)
        
        # Add emotion label if predictions available
        if predictions and i < len(predictions):
            emotion = LABELS[predictions[i]]
            draw.text((square_x1, square_y1-20), f"{emotion} ({confidence:.2f})", fill="red")
        else:
            draw.text((square_x1, square_y1-20), f"Face ({confidence:.2f})", fill="red")
    
    return image

# ---- Streamlit UI ----
# Input method selection
input_method = st.radio("Choose input method:", ["Upload Image", "Use Webcam"])

if input_method == "Upload Image":
    uploaded_file = st.file_uploader('Upload a face image', type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_container_width=True)
        
        # Load models
        models, model_names, weights = load_models()
        yolo_model = load_yolo_model()
        
        # Detect faces
        faces = detect_faces(img, yolo_model)
        
        if faces:
            st.success(f"Detected {len(faces)} face(s)")
            
            # Process each detected face
            predictions = []
            for i, face_bbox in enumerate(faces):
                face_img = crop_face(img, face_bbox)
                ensembled_probs, all_model_probs, ensembled_pred = process_and_predict(face_img, models, model_names, weights)
                predictions.append(ensembled_pred)
                
                st.subheader(f'Face {i+1} - Ensembled Prediction')
                st.write(f"**Predicted Emotion:** {LABELS[ensembled_pred]}")
                st.bar_chart(ensembled_probs)
                
                # Show cropped face
                st.image(face_img, caption=f'Detected Face {i+1}', width=200)
                
                # Individual model predictions
                st.write('Individual Model Predictions:')
                for j, probs in enumerate(all_model_probs):
                    pred = int(np.argmax(probs))
                    st.write(f"- {model_names[j]}: {LABELS[pred]}")
            
            # Show image with bounding boxes
            img_with_boxes = draw_face_boxes(img.copy(), faces, predictions)
            st.image(img_with_boxes, caption='Detected Faces with Emotions', use_container_width=True)
            
        else:
            st.warning("No faces detected in the image. Try uploading a clearer image with visible faces.")

elif input_method == "Use Webcam":
    st.write("Click 'Start' to capture from webcam")
    
    # Webcam capture
    if st.button("Start Webcam"):
        # Load models
        models, model_names, weights = load_models()
        yolo_model = load_yolo_model()
        
        # Create placeholders for real-time updates
        frame_placeholder = st.empty()
        results_placeholder = st.empty()
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open webcam")
        else:
            stop_button = st.button("Stop Webcam")
            
            while not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Detect faces
                faces = detect_faces(pil_image, yolo_model)
                
                if faces:
                    predictions = []
                    emotion_results = []
                    
                    for face_bbox in faces:
                        face_img = crop_face(pil_image, face_bbox)
                        ensembled_probs, all_model_probs, ensembled_pred = process_and_predict(face_img, models, model_names, weights)
                        predictions.append(ensembled_pred)
                        emotion_results.append({
                            'emotion': LABELS[ensembled_pred],
                            'confidence': np.max(ensembled_probs)
                        })
                    
                    # Draw bounding boxes with emotions
                    pil_image = draw_face_boxes(pil_image, faces, predictions)
                    
                    # Display results
                    with results_placeholder.container():
                        st.write("**Real-time Emotion Detection:**")
                        for i, result in enumerate(emotion_results):
                            st.write(f"Face {i+1}: {result['emotion']} (confidence: {result['confidence']:.2f})")
                
                # Display frame
                frame_placeholder.image(pil_image, channels="RGB", use_container_width=True)
                
                # Small delay to prevent overwhelming the system
                import time
                time.sleep(0.1)
            
            cap.release()
