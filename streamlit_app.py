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
import time
import pandas as pd

# ---- Page Configuration ----
st.set_page_config(
    page_title="Facial Expression Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- User Instructions ----
st.title('😊 Facial Expression Recognition (Ensemble)')
st.markdown('**Upload a face image or use webcam to identify emotions using an ensemble of deep learning models.**')

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

# ---- Performance Optimizations ----
@st.cache_data
def get_device():
    """Get the best available device for inference"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# ---- Image Preprocessing (Fixed to match notebook) ----
def preprocess_image(img):
    """
    Preprocess image exactly as done in the training notebook
    This ensures consistency with model training
    """
    # Convert to grayscale and resize to 48x48
    img = ImageOps.grayscale(img)
    img = img.resize((48, 48))
    
    # Convert to numpy array and reshape to match training format
    img_array = np.array(img).astype(np.uint8)
    img_array = img_array.reshape(48, 48, 1)  # (48, 48, 1)
    
    # Apply the same transforms as in notebook
    transform = T.Compose([
        T.ToPILImage(),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),  # Same normalization as training
    ])
    
    img_tensor = transform(img_array)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension (1, 1, 48, 48)
    return img_tensor

# ---- YOLO Face Detection (Optimized) ----
@st.cache_resource
def load_yolo_model():
    """Load YOLO model for face detection with caching"""
    try:
        model = YOLO("yolov12n-face.pt")
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None

def detect_faces(image, yolo_model, confidence_threshold=0.5):
    """Detect faces in image using YOLO with optimized parameters"""
    if yolo_model is None:
        return []
    
    try:
        # Optimize YOLO inference
        results = yolo_model(image, conf=confidence_threshold, verbose=False)
        faces = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Filter for high confidence detections
                    if confidence > confidence_threshold:
                        faces.append((int(x1), int(y1), int(x2), int(y2), confidence))
        
        return faces
    except Exception as e:
        st.error(f"Face detection error: {e}")
        return []

def crop_face(image, bbox):
    """Crop face from image using bounding box with optimal padding"""
    x1, y1, x2, y2, _ = bbox
    
    # Calculate current face dimensions
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Add padding (40% on each side for better face context)
    padding_factor = 0.4
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

# ---- Model Loading (Optimized) ----
@st.cache_resource
def load_models():
    """Load ensemble models with proper device placement and caching"""
    try:
        from model import EfficientNetB0, to_device
        
        device = get_device()
        st.info(f"Using device: {device}")
        
        models = []
        model_names = []
        weights = []
        
        # Load model 1: CRP augmentation model
        model_eff_b0 = to_device(EfficientNetB0(7, 1), device)
        model_eff_b0.load_state_dict(
            torch.load('model_efficientnetb0_v2_tuned_2.pth', map_location=device)
        )
        model_eff_b0.eval()
        models.append(model_eff_b0)
        model_names.append('EfficientNetB0_v2_tuned_2 (CRP)')
        weights.append(0.45)
        
        # Load model 2: MixUp/CutMix model
        model_eff_b0_2 = to_device(EfficientNetB0(7, 1), device)
        model_eff_b0_2.load_state_dict(
            torch.load('model_efficientnetb0_v5_tuned.pth', map_location=device)
        )
        model_eff_b0_2.eval()
        models.append(model_eff_b0_2)
        model_names.append('EfficientNetB0_v5_tuned (MixUp/CutMix)')
        weights.append(0.55)
        
        return models, model_names, weights
        
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return [], [], []

# ---- Ensemble Prediction (Optimized) ----
def weighted_average_probs(models, img_tensor, weights, device):
    """Weighted average ensemble with proper device handling"""
    probs_list = []
    
    with torch.no_grad():
        for model in models:
            # Ensure input is on the same device as model
            img_tensor = img_tensor.to(device)
            
            # Get model predictions
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)
            probs_list.append(probs.cpu().numpy())
    
    # Stack and weight the probabilities
    probs_array = np.stack(probs_list, axis=0)  # (num_models, 1, num_classes)
    weights = np.array(weights).reshape(-1, 1, 1)
    weighted_probs = (probs_array * weights).sum(axis=0)  # (1, num_classes)
    
    return weighted_probs[0], probs_list

def process_and_predict(face_image, models, model_names, weights, device):
    """Process face image and make predictions with proper error handling"""
    try:
        img_tensor = preprocess_image(face_image)
        ensembled_probs, all_model_probs = weighted_average_probs(
            models, img_tensor, weights, device
        )
        ensembled_pred = int(np.argmax(ensembled_probs))
        return ensembled_probs, all_model_probs, ensembled_pred
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

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
        draw.rectangle([square_x1, square_y1, square_x2, square_y2], outline="red", width=3)
        
        # Add emotion label if predictions available
        if predictions and i < len(predictions):
            emotion = LABELS[predictions[i]]
            # Create a background for better text visibility
            text_bbox = draw.textbbox((square_x1, square_y1-25), f"{emotion} ({confidence:.2f})")
            draw.rectangle(text_bbox, fill="red")
            draw.text((square_x1, square_y1-25), f"{emotion} ({confidence:.2f})", fill="white")
        else:
            text_bbox = draw.textbbox((square_x1, square_y1-25), f"Face ({confidence:.2f})")
            draw.rectangle(text_bbox, fill="red")
            draw.text((square_x1, square_y1-25), f"Face ({confidence:.2f})", fill="white")
    
    return image

# ---- Performance Metrics ----
def display_performance_metrics():
    """Display model performance metrics"""
    st.sidebar.markdown("## 📊 Model Performance")
    st.sidebar.markdown(f"**Final Accuracy:** 68.74%")
    st.sidebar.markdown(f"**GFLOPs:** 0.046")
    st.sidebar.markdown("**Ensemble:** 2 EfficientNet-B0 models")
    st.sidebar.markdown("- CRP Augmentation (45%)")
    st.sidebar.markdown("- MixUp/CutMix (55%)")

# ---- Main Application ----
def main():
    # Display performance metrics in sidebar
    display_performance_metrics()
    
    # Load models and YOLO
    with st.spinner("Loading models..."):
        models, model_names, weights = load_models()
        yolo_model = load_yolo_model()
    
    if not models:
        st.error("Failed to load models. Please check if model files exist.")
        return
    
    device = get_device()
    
    # Input method selection
    input_method = st.radio("Choose input method:", ["Upload Image", "Use Webcam"])
    
    if input_method == "Upload Image":
        uploaded_file = st.file_uploader('Upload a face image', type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            # Display uploaded image
            col1, col2 = st.columns([2, 1])
            
            with col1:
                img = Image.open(uploaded_file)
                st.image(img, caption='Uploaded Image', use_container_width=True)
            
            with col2:
                st.markdown("### 📋 Processing Steps")
                st.markdown("1. **Face Detection** - YOLO model")
                st.markdown("2. **Face Cropping** - Square padding")
                st.markdown("3. **Preprocessing** - 48x48 grayscale")
                st.markdown("4. **Ensemble Prediction** - 2 models")
                st.markdown("5. **Result Display** - Emotion + confidence")
            
            # Detect faces
            with st.spinner("Detecting faces..."):
                faces = detect_faces(img, yolo_model)
            
            if faces:
                st.success(f"✅ Detected {len(faces)} face(s)")
                
                # Process each detected face
                for i, face_bbox in enumerate(faces):
                    st.markdown(f"---")
                    st.subheader(f'😊 Face {i+1} - Emotion Analysis')
                    
                    # Create columns for face display and results
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        face_img = crop_face(img, face_bbox)
                        st.image(face_img, caption=f'Detected Face {i+1}', width=200)
                    
                    with col2:
                        # Get predictions
                        ensembled_probs, all_model_probs, ensembled_pred = process_and_predict(
                            face_img, models, model_names, weights, device
                        )
                        
                        if ensembled_probs is not None:
                            # Get top 3 emotions with their probabilities
                            emotion_probs = list(zip(LABELS.values(), ensembled_probs))
                            emotion_probs.sort(key=lambda x: x[1], reverse=True)
                            top_3_emotions = emotion_probs[:3]
                            
                            # Display top 3 emotions
                            st.markdown("**🏆 Top 3 Emotions:**")
                            for i, (emotion, prob) in enumerate(top_3_emotions):
                                if i == 0:
                                    st.markdown(f"🥇 **{emotion}:** {prob:.3f}")
                                elif i == 1:
                                    st.markdown(f"🥈 **{emotion}:** {prob:.3f}")
                                else:
                                    st.markdown(f"🥉 **{emotion}:** {prob:.3f}")
                            
                            # Display probability chart
                            prob_df = pd.DataFrame({
                                'Emotion': list(LABELS.values()),
                                'Probability': ensembled_probs
                            })
                            st.bar_chart(prob_df.set_index('Emotion'))
                            
                            # Individual model predictions
                            st.markdown("**🔍 Individual Model Predictions:**")
                            for j, probs in enumerate(all_model_probs):
                                pred = int(np.argmax(probs))
                                confidence = np.max(probs)
                                st.markdown(f"- **{model_names[j]}:** {LABELS[pred]} ({confidence:.3f})")
                
                # Show image with bounding boxes
                st.markdown("---")
                st.subheader("📍 Detection Results")
                img_with_boxes = draw_face_boxes(img.copy(), faces, predictions)
                st.image(img_with_boxes, caption='Detected Faces with Emotions', use_container_width=True)
                
            else:
                st.warning("⚠️ No faces detected in the image. Try uploading a clearer image with visible faces.")

    elif input_method == "Use Webcam":
        st.markdown("### 📹 Real-time Webcam Emotion Detection")
        st.markdown("Click 'Start Webcam' to begin real-time emotion detection.")
        
        # Webcam capture
        if st.button("🎥 Start Webcam", type="primary"):
            # Load models if not already loaded
            if not models:
                st.error("Models not loaded. Please try again.")
                return
            
            # Create placeholders for real-time updates
            frame_placeholder = st.empty()
            results_placeholder = st.empty()
            fps_placeholder = st.sidebar.empty()
            
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Could not open webcam")
            else:
                st.success("✅ Webcam started successfully")
                
                # Performance optimization for webcam
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)  # Reduce FPS for better performance
                
                stop_button = st.button("⏹️ Stop Webcam")
                
                frame_count = 0
                start_time = time.time()
                
                while not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Failed to capture frame")
                        break
                    
                    # Process every 3rd frame for better performance
                    frame_count += 1
                    if frame_count % 3 != 0:
                        continue
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Detect faces
                    faces = detect_faces(pil_image, yolo_model, confidence_threshold=0.6)
                    
                    if faces:
                        predictions = []
                        emotion_results = []
                        
                        for face_bbox in faces:
                            face_img = crop_face(pil_image, face_bbox)
                            ensembled_probs, all_model_probs, ensembled_pred = process_and_predict(
                                face_img, models, model_names, weights, device
                            )
                            
                            if ensembled_probs is not None:
                                predictions.append(ensembled_pred)
                                emotion_results.append({
                                    'emotion': LABELS[ensembled_pred],
                                    'confidence': np.max(ensembled_probs),
                                    'probabilities': ensembled_probs
                                })
                        
                        # Draw bounding boxes with emotions
                        if predictions:
                            pil_image = draw_face_boxes(pil_image, faces, predictions)
                        
                        # Display results
                        with results_placeholder.container():
                            st.markdown("**🎯 Real-time Emotion Detection:**")
                            for i, result in enumerate(emotion_results):
                                # Get top 3 emotions for this face
                                face_probs = result['probabilities']
                                emotion_probs = list(zip(LABELS.values(), face_probs))
                                emotion_probs.sort(key=lambda x: x[1], reverse=True)
                                top_3_emotions = emotion_probs[:3]
                                
                                st.markdown(f"**Face {i+1}:**")
                                for j, (emotion, prob) in enumerate(top_3_emotions):
                                    if j == 0:
                                        st.markdown(f"  🥇 {emotion}: {prob:.3f}")
                                    elif j == 1:
                                        st.markdown(f"  🥈 {emotion}: {prob:.3f}")
                                    else:
                                        st.markdown(f"  🥉 {emotion}: {prob:.3f}")
                    
                    # Display frame
                    frame_placeholder.image(pil_image, channels="RGB", use_container_width=True)
                    
                    # Calculate and display FPS (update in place)
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        fps = frame_count / elapsed_time
                        fps_placeholder.metric("🎯 Real-time FPS", f"{fps:.1f}")
                    
                    # Small delay to prevent overwhelming the system
                    time.sleep(0.05)  # Reduced delay for better responsiveness
                
                cap.release()
                st.success("✅ Webcam stopped")

if __name__ == "__main__":
    main()
