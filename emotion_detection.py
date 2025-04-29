import cv2
import numpy as np
import os
import requests
from tqdm import tqdm
from datetime import datetime
import time
import tensorflow as tf
import mediapipe as mp
from collections import deque
import threading
import random
from deepface import DeepFace

class EnhancedEmotionDetector:
    """Advanced emotion detection system with improved accuracy and efficiency"""
    
    def __init__(self):
        # Model paths
        self.emotion_model_path = 'emotion_model.onnx'
        self.face_model_path = 'face_model.onnx'
        
        # Initialize emotion labels and emojis - SIMPLIFIED SET
        self.emotion_emojis = {
            'Happy': '😊',
            'Sad': '😢',
            'Angry': '😠',
            'Neutral': '😐'
        }
        self.emotion_labels = list(self.emotion_emojis.keys())
        
        # Color coding for emotions
        self.emotion_colors = {
            'Happy': (0, 255, 0),      # Green for happy
            'Neutral': (0, 0, 0),      # Black for neutral
            'Sad': (255, 0, 0),        # Blue for sad (BGR format)
            'Angry': (0, 0, 255)       # Red for angry
        }
        
        # Initialize tracking and smoothing
        self.emotion_history = deque(maxlen=10)  # For temporal smoothing
        self.face_trackers = {}  # For face tracking
        self.next_face_id = 0
        
        # MediaPipe for more precise face detection and landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Performance metrics
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Background model loading flag
        self.models_loaded = False
        self.loading_thread = None
        
        # Setup models and components
        self.setup()
    
    def setup(self):
        """Initialize and load all detection models"""
        try:
            # Start model loading in background
            self.loading_thread = threading.Thread(target=self.load_models)
            self.loading_thread.daemon = True
            self.loading_thread.start()
            
            # Fallback face detector for cases where MediaPipe might fail
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            print("Initializing models in background...")
            
        except Exception as e:
            print(f"Error during setup: {e}")
            raise
    
    def load_models(self):
        """Load models in background thread for better UI responsiveness"""
        try:
            # Setup MediaPipe components for better face detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                min_detection_confidence=0.5,  # Lower threshold for better detection
                model_selection=1  # Use full range model (better for different distances)
            )
            
            # Create face mesh with full landmarks for better emotion detection
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,  # Focus on one face for better performance
                min_detection_confidence=0.5,  # Lower threshold for better detection
                min_tracking_confidence=0.5,  # Lower threshold for better tracking
                refine_landmarks=True  # Enhanced landmark accuracy
            )
            
            # Using MediaPipe Face Mesh for emotion detection instead of ONNX
            # This avoids the ONNX model compatibility issues
            print("Using MediaPipe for emotion detection (more reliable)")
            self.models_loaded = True
            print("Models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()
    
    def download_emotion_model(self):
        """Download the pre-trained emotion detection model (fallback method)"""
        # Since we're using MediaPipe instead, this is a fallback
        print("Using built-in MediaPipe models instead of downloading")
        pass
    
    def preprocess_face(self, face):
        """Advanced preprocessing pipeline for better emotion detection"""
        # Convert to grayscale if not already
        if len(face.shape) == 3:
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face
            
        # Apply histogram equalization for better contrast
        face_equalized = cv2.equalizeHist(face_gray)
        
        # Apply Gaussian blur to reduce noise
        face_blurred = cv2.GaussianBlur(face_equalized, (3, 3), 0)
        
        # Resize to required input dimensions
        face_resized = cv2.resize(face_blurred, (64, 64))
        
        # Normalize pixel values
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        # Prepare for model input
        # Reshape to match model input shape (adding batch and channel dimensions)
        model_input = np.expand_dims(np.expand_dims(face_normalized, axis=0), axis=0)
        
        return model_input
    
    def detect_faces_mediapipe(self, frame):
        """Detect faces using MediaPipe for higher accuracy"""
        if not self.models_loaded:
            return []  # Return empty if models aren't loaded yet
            
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]
        
        # Process the image
        results = self.face_detection.process(rgb_frame)
        faces = []
        
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * frame_width)
                y = int(bboxC.ymin * frame_height)
                w = int(bboxC.width * frame_width)
                h = int(bboxC.height * frame_height)
                
                # Add some margin for better emotion detection
                x_margin = int(w * 0.1)
                y_margin = int(h * 0.1)
                x = max(0, x - x_margin)
                y = max(0, y - y_margin)
                w = min(frame_width - x, w + 2 * x_margin)
                h = min(frame_height - y, h + 2 * y_margin)
                
                # Skip if face dimensions are invalid
                if w <= 0 or h <= 0:
                    continue
                    
                # Get detection confidence
                confidence = detection.score[0]
                faces.append((x, y, w, h, confidence))
        
        return faces
    
    def detect_faces_cascade(self, frame):
        """Fallback face detection using Haar cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Convert to same format as MediaPipe detection
        return [(x, y, w, h, 0.8) for (x, y, w, h) in faces]
    
    def detect_landmarks(self, frame, face_rect):
        """Detect facial landmarks for more accurate emotion analysis"""
        if not self.models_loaded:
            return None  # Return None if models aren't loaded yet
            
        x, y, w, h, _ = face_rect
        
        # Make sure coordinates are valid
        x, y = max(0, x), max(0, y)
        frame_h, frame_w = frame.shape[:2]
        w = min(frame_w - x, w)
        h = min(frame_h - y, h)
        
        if w <= 0 or h <= 0:
            return None
            
        # Extract face ROI
        try:
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0:
                return None
                
            # Convert to RGB for MediaPipe
            rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            
            # Process with Face Mesh
            results = self.face_mesh.process(rgb_roi)
            
            landmarks = []
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Convert normalized landmarks to pixel coordinates
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        # Map normalized coordinates to ROI dimensions
                        px = min(int(landmark.x * w), w - 1)
                        py = min(int(landmark.y * h), h - 1)
                        # Add offset to get coordinates in original frame
                        landmarks.append((x + px, y + py, idx))
                        
            return landmarks if landmarks else None
        except Exception as e:
            print(f"Error detecting landmarks: {e}")
            return None
    
    def track_faces(self, faces, frame_width, frame_height):
        """Track faces across frames for stable emotion detection"""
        current_faces = {}
        
        for face in faces:
            x, y, w, h, conf = face
            
            # Center point of the face
            face_center = (x + w // 2, y + h // 2)
            
            # Check if this face overlaps with any tracked face
            matched = False
            for face_id, tracked_face in list(self.face_trackers.items()):
                tx, ty, tw, th = tracked_face['rect']
                tracked_center = (tx + tw // 2, ty + th // 2)
                
                # Calculate distance between centers
                distance = np.sqrt((face_center[0] - tracked_center[0])**2 + 
                                  (face_center[1] - tracked_center[1])**2)
                
                # If centers are close enough, consider it the same face
                if distance < (w + tw) // 4:
                    # Update tracker with new position
                    # Apply EMA smoothing for stable tracking
                    alpha = 0.8  # Smoothing factor
                    new_x = int(alpha * x + (1 - alpha) * tx) 
                    new_y = int(alpha * y + (1 - alpha) * ty)
                    new_w = int(alpha * w + (1 - alpha) * tw)
                    new_h = int(alpha * h + (1 - alpha) * th)
                    
                    # Update tracked face
                    self.face_trackers[face_id]['rect'] = (new_x, new_y, new_w, new_h)
                    self.face_trackers[face_id]['ttl'] = 10  # Reset time-to-live
                    current_faces[face_id] = self.face_trackers[face_id]
                    matched = True
                    break
            
            # If no match found, create new tracker
            if not matched:
                face_id = self.next_face_id
                self.next_face_id += 1
                self.face_trackers[face_id] = {
                    'rect': (x, y, w, h),
                    'ttl': 10,  # Time to live counter
                    'emotion_history': [None] * 5  # For emotion smoothing
                }
                current_faces[face_id] = self.face_trackers[face_id]
        
        # Update TTL for all trackers and remove expired ones
        for face_id in list(self.face_trackers.keys()):
            if face_id not in current_faces:
                self.face_trackers[face_id]['ttl'] -= 1
                if self.face_trackers[face_id]['ttl'] <= 0:
                    del self.face_trackers[face_id]
                else:
                    current_faces[face_id] = self.face_trackers[face_id]
        
        return current_faces
    
    def predict_emotion(self, face_img):
        """Predict emotion from face image with improved accuracy using facial landmarks and geometry"""
        h, w = face_img.shape[:2]
        
        # Convert to grayscale for better processing
        if len(face_img.shape) == 3:
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_img
        
        # Initialize default probabilities
        emotion_probs = {emotion: 0.05 for emotion in self.emotion_labels}
        emotion_probs['Neutral'] = 0.5  # Default to neutral
        
        # Debug info
        debug_info = {}
        
        # Detect face landmarks if MediaPipe is loaded
        if self.models_loaded and hasattr(self, 'face_mesh'):
            try:
                # Preprocess for MediaPipe
                rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                # Get facial landmarks
                results = self.face_mesh.process(rgb_img)
                
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    
                    # Extract key facial features for emotion analysis
                    
                    # Eyes features
                    left_eye = []
                    right_eye = []
                    for i in [33, 133, 173, 157, 158, 159]:  # Left eye landmarks
                        if i < len(landmarks.landmark):
                            left_eye.append((landmarks.landmark[i].x, landmarks.landmark[i].y))
                    
                    for i in [362, 263, 466, 388, 387, 386]:  # Right eye landmarks
                        if i < len(landmarks.landmark):
                            right_eye.append((landmarks.landmark[i].x, landmarks.landmark[i].y))
                    
                    # Mouth features
                    mouth_top = []
                    mouth_bottom = []
                    for i in [13, 14, 17, 37, 39, 40, 61, 84, 181, 91, 146]:  # Top lip
                        if i < len(landmarks.landmark):
                            mouth_top.append((landmarks.landmark[i].x, landmarks.landmark[i].y))
                    
                    for i in [17, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314]:  # Bottom lip
                        if i < len(landmarks.landmark):
                            mouth_bottom.append((landmarks.landmark[i].x, landmarks.landmark[i].y))
                    
                    # Eyebrows
                    left_eyebrow = []
                    right_eyebrow = []
                    for i in [70, 63, 105, 66, 107]:  # Left eyebrow
                        if i < len(landmarks.landmark):
                            left_eyebrow.append((landmarks.landmark[i].x, landmarks.landmark[i].y))
                    
                    for i in [336, 296, 334, 293, 300]:  # Right eyebrow
                        if i < len(landmarks.landmark):
                            right_eyebrow.append((landmarks.landmark[i].x, landmarks.landmark[i].y))
                    
                    # Cheeks and smile lines (these are crucial for happy detection)
                    left_cheek = []
                    right_cheek = []
                    for i in [206, 187, 147, 123, 116, 143, 156, 70]:  # Left cheek
                        if i < len(landmarks.landmark):
                            left_cheek.append((landmarks.landmark[i].x, landmarks.landmark[i].y))
                    
                    for i in [427, 411, 376, 352, 345, 372, 383, 300]:  # Right cheek
                        if i < len(landmarks.landmark):
                            right_cheek.append((landmarks.landmark[i].x, landmarks.landmark[i].y))
                    
                    # Check if we have enough landmarks for analysis
                    if left_eye and right_eye and mouth_top and mouth_bottom:
                        # Calculate eye aspect ratio (for blink/surprise detection)
                        def eye_aspect_ratio(eye):
                            if len(eye) >= 6:
                                # Vertical eye distance 
                                A = np.sqrt((eye[1][0] - eye[5][0])**2 + (eye[1][1] - eye[5][1])**2)
                                B = np.sqrt((eye[2][0] - eye[4][0])**2 + (eye[2][1] - eye[4][1])**2)
                                # Horizontal eye distance
                                C = np.sqrt((eye[0][0] - eye[3][0])**2 + (eye[0][1] - eye[3][1])**2)
                                # Eye aspect ratio
                                return (A + B) / (2.0 * C) if C > 0 else 0
                            return 0
                        
                        # Calculate mouth height and width
                        def mouth_metrics(top, bottom):
                            if len(top) >= 3 and len(bottom) >= 3:
                                # Get average y positions
                                top_y = sum(y for _, y in top) / len(top)
                                bottom_y = sum(y for _, y in bottom) / len(bottom)
                                # Get leftmost and rightmost points
                                all_x = [x for x, _ in top + bottom]
                                left_x = min(all_x) if all_x else 0
                                right_x = max(all_x) if all_x else 0
                                
                                # Calculate height and width
                                height = bottom_y - top_y
                                width = right_x - left_x
                                
                                # Calculate ratio: higher for smiles
                                ratio = width / height if height > 0 else 0
                                return height, width, ratio
                            return 0, 0, 0
                        
                        # Calculate smile curvature
                        def smile_curvature(top, bottom):
                            if len(top) >= 5 and len(bottom) >= 5:
                                # Average position of middle points
                                middle_top = top[len(top)//2]
                                middle_bottom = bottom[len(bottom)//2]
                                
                                # Average of left corner and right corner
                                left_corner = (top[0][1] + bottom[0][1]) / 2
                                right_corner = (top[-1][1] + bottom[-1][1]) / 2
                                
                                # Average corner height
                                corner_avg = (left_corner + right_corner) / 2
                                
                                # Curvature: negative for smile (corners higher than middle)
                                curve_top = middle_top[1] - corner_avg
                                curve_bottom = middle_bottom[1] - corner_avg
                                
                                return curve_top, curve_bottom
                            return 0, 0
                        
                        # Calculate eyebrow position relative to eyes
                        def eyebrow_position(eyebrow, eye):
                            if len(eyebrow) >= 3 and len(eye) >= 3:
                                eyebrow_y = sum(y for _, y in eyebrow) / len(eyebrow)
                                eye_y = sum(y for _, y in eye) / len(eye)
                                return eye_y - eyebrow_y  # Higher value means raised eyebrows
                            return 0
                        
                        # Calculate the metrics
                        left_ear = eye_aspect_ratio(left_eye)
                        right_ear = eye_aspect_ratio(right_eye)
                        
                        mouth_height, mouth_width, mouth_ratio = mouth_metrics(mouth_top, mouth_bottom)
                        curve_top, curve_bottom = smile_curvature(mouth_top, mouth_bottom)
                        
                        left_brow_pos = eyebrow_position(left_eyebrow, left_eye)
                        right_brow_pos = eyebrow_position(right_eyebrow, right_eye)
                        
                        # Average eye aspect ratio and brow position
                        ear = (left_ear + right_ear) / 2
                        brow_pos = (left_brow_pos + right_brow_pos) / 2
                        
                        # Save metrics for debugging
                        debug_info = {
                            'ear': ear,
                            'mouth_ratio': mouth_ratio,
                            'curve_top': curve_top,
                            'curve_bottom': curve_bottom,
                            'brow_pos': brow_pos
                        }
                        
                        # IMPROVED Feature-based emotion classification
                        
                        # Happy: curved smile shape (corners up), wider mouth ratio
                        # Check for smile - curved mouth and good width/height ratio
                        # More balanced detection
                        is_smiling = curve_top < -0.01 and mouth_ratio > 2.2
                        
                        if is_smiling:
                            emotion_probs['Happy'] = 0.8
                            emotion_probs['Neutral'] = 0.15
                            # Reduce others significantly
                            for e in ['Sad', 'Angry']:
                                emotion_probs[e] = 0.016
                        
                        # Sad: Droopy mouth corners, slight frown, lower eyebrows
                        elif curve_bottom > 0.01 and brow_pos < 0.04 and mouth_ratio < 2.5:
                            emotion_probs['Sad'] = 0.75
                            emotion_probs['Neutral'] = 0.15
                            # Reduce happy
                            emotion_probs['Happy'] = 0.01
                            
                        # Angry: Lowered eyebrows, narrowed eyes, tight mouth - more aggressive detection
                        elif brow_pos < 0.05 or ear < 0.3:
                            emotion_probs['Angry'] = 0.9
                            emotion_probs['Neutral'] = 0.05
                            emotion_probs['Happy'] = 0.01
                            emotion_probs['Sad'] = 0.02
                            
                        # Neutral: Average metrics, no strong indicators
                        else:
                            emotion_probs['Neutral'] = 0.75
                            # Distribute remaining prob among others
                            for e in ['Happy', 'Sad']:
                                emotion_probs[e] = 0.06
                            
            except Exception as e:
                print(f"Error in landmark analysis: {e}")
        
        # If we don't have good landmark data, try image-based analysis
        if max(emotion_probs.values()) <= 0.2:
            # Use histogram of oriented gradients for basic emotion features
            try:
                # Resize for consistent processing
                resized_face = cv2.resize(gray_face, (64, 64))
                
                # Enhance contrast
                equalized_face = cv2.equalizeHist(resized_face)
                
                # Simple edge detection for facial features
                edges = cv2.Canny(equalized_face, 100, 200)
                
                # Count edge pixels in regions (forehead, eyes, mouth)
                face_h, face_w = edges.shape
                
                # Eye region
                eye_region = edges[int(face_h*0.2):int(face_h*0.5), :]
                eye_intensity = np.sum(eye_region) / (eye_region.size * 255)
                
                # Mouth region
                mouth_region = edges[int(face_h*0.6):int(face_h*0.9), :]
                mouth_intensity = np.sum(mouth_region) / (mouth_region.size * 255)
                
                # Midface (nose, cheeks)
                mid_region = edges[int(face_h*0.3):int(face_h*0.7), int(face_w*0.3):int(face_w*0.7)]
                mid_intensity = np.sum(mid_region) / (mid_region.size * 255)
                
                # Brightness values for different regions
                forehead_brightness = np.mean(resized_face[0:int(face_h*0.2), :])
                eye_brightness = np.mean(resized_face[int(face_h*0.2):int(face_h*0.5), :])
                mouth_brightness = np.mean(resized_face[int(face_h*0.6):int(face_h*0.9), :])
                
                # Look for smile pattern - high intensity in mouth region with specific pattern
                smile_detected = False
                
                # Check for smile - higher intensity in mouth corners than center
                left_mouth = mouth_region[:, 0:int(face_w*0.4)]
                right_mouth = mouth_region[:, int(face_w*0.6):]
                center_mouth = mouth_region[:, int(face_w*0.4):int(face_w*0.6)]
                
                left_intensity = np.sum(left_mouth) / (left_mouth.size * 255) if left_mouth.size > 0 else 0
                right_intensity = np.sum(right_mouth) / (right_mouth.size * 255) if right_mouth.size > 0 else 0
                center_intensity = np.sum(center_mouth) / (center_mouth.size * 255) if center_mouth.size > 0 else 0
                
                # Smile typically has higher intensity at corners
                if left_intensity > 0.08 and right_intensity > 0.08 and center_intensity < max(left_intensity, right_intensity):
                    smile_detected = True
                
                # Happy detection: smile pattern or high mouth intensity with bright mouth
                if smile_detected and mouth_brightness > 100:
                    emotion_probs['Happy'] = 0.8
                    emotion_probs['Neutral'] = 0.15
                    for e in ['Sad']:
                        emotion_probs[e] = 0.016
                
                # Sad detection: low mouth intensity, low overall brightness
                elif mouth_intensity < 0.1 and mouth_brightness < 100:
                    emotion_probs['Sad'] = 0.7
                
                # Fear detection: high eye intensity with raised eyebrows (more intensity in upper eye region)
                elif eye_intensity > 0.2 and np.mean(edges[int(face_h*0.2):int(face_h*0.3), :]) > 0.15:
                    emotion_probs['Sad'] = 0.7
                
                # Angry detection: high intensity in all regions, low brightness
                elif eye_intensity > 0.1 or (mouth_intensity > 0.05 and forehead_brightness < 130):
                    emotion_probs['Angry'] = 0.9
                    emotion_probs['Neutral'] = 0.05
                    emotion_probs['Happy'] = 0.01
                    emotion_probs['Sad'] = 0.02
                
                # Neutral otherwise
                else:
                    emotion_probs['Neutral'] = 0.7
                    
            except Exception as e:
                print(f"Error in image analysis: {e}")
                # Default to neutral if all else fails
                emotion_probs['Neutral'] = 0.6
        
        # Normalize probabilities
        total = sum(emotion_probs.values())
        if total > 0:
            for emotion in emotion_probs:
                emotion_probs[emotion] /= total
        
        # Final adjustment to avoid misclassifying smiles as fear
        # If happy and fear are both high, favor happy
        if emotion_probs['Happy'] > 0.25 and emotion_probs['Sad'] > 0.25:
            emotion_probs['Happy'] += emotion_probs['Sad'] * 0.5
            emotion_probs['Sad'] *= 0.5
            
            # Renormalize
            total = sum(emotion_probs.values())
            for emotion in emotion_probs:
                emotion_probs[emotion] /= total
        
        # Get the emotion with the highest probability
        max_emotion = max(emotion_probs, key=emotion_probs.get)
        confidence = emotion_probs[max_emotion] * 100
        
        # Print debug info
        print(f"Emotion: {max_emotion}, Confidence: {confidence:.1f}%, Metrics: {debug_info}")
        
        return max_emotion, confidence, emotion_probs
    
    def smooth_emotion(self, face_id, emotion, confidence, emotion_probs):
        """Apply temporal smoothing to emotion predictions"""
        # Get emotion history for this face
        if 'emotion_history' not in self.face_trackers[face_id]:
            self.face_trackers[face_id]['emotion_history'] = []
            
        emotion_history = self.face_trackers[face_id]['emotion_history']
        
        # Add current emotion to history
        emotion_history.append((emotion, confidence, emotion_probs))
        
        # Only keep the latest predictions
        max_history = 5
        if len(emotion_history) > max_history:
            emotion_history = emotion_history[-max_history:]
        
        # Update the emotion history
        self.face_trackers[face_id]['emotion_history'] = emotion_history
        
        # If history is too short, return current emotion
        if len(emotion_history) < 2:
            return emotion, confidence, emotion_probs
        
        # Calculate weighted average of emotions
        weights = [0.2, 0.3, 0.5, 0.7, 1.0]  # More weight to recent predictions
        weights = weights[-len(emotion_history):]
        
        # Accumulate emotion probabilities
        accum_probs = {e: 0.0 for e in self.emotion_labels}
        
        for i, (e, c, probs) in enumerate(emotion_history):
            if probs:  # Check if probs exists
                for emotion_name, prob in probs.items():
                    if emotion_name in accum_probs:
                        accum_probs[emotion_name] += prob * weights[i]
        
        # Normalize accumulated probabilities
        total = sum(accum_probs.values())
        if total > 0:
            for emotion_name in accum_probs:
                accum_probs[emotion_name] /= total
        
        # Get smoothed emotion with highest probability
        smoothed_emotion = max(accum_probs, key=accum_probs.get)
        smoothed_confidence = accum_probs[smoothed_emotion] * 100
        
        return smoothed_emotion, smoothed_confidence, accum_probs
    
    def draw_landmarks(self, frame, landmarks, color=(0, 255, 0)):
        """Draw facial landmarks on the frame"""
        if not landmarks:
            return
            
        # Group landmarks by facial feature for better visualization
        feature_groups = {
            'Contour': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152],
            'Left Eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'Right Eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'Left Eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'Right Eyebrow': [336, 296, 334, 293, 300, 285, 295, 282, 283, 276],
            'Nose': [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 97, 326, 327],
            'Mouth': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
        }
        
        # Specify distinct colors for each feature group
        feature_colors = {
            'Contour': (100, 100, 100),  # Gray
            'Left Eye': (0, 255, 0),     # Green
            'Right Eye': (0, 255, 0),    # Green
            'Left Eyebrow': (255, 0, 0), # Blue
            'Right Eyebrow': (255, 0, 0),# Blue
            'Nose': (0, 165, 255),       # Orange
            'Mouth': (0, 0, 255)         # Red
        }
        
        # Create dictionary to look up landmark indices
        landmark_dict = {}
        for lm in landmarks:
            if len(lm) == 3:  # If idx is included
                x, y, idx = lm
                landmark_dict[idx] = (x, y)
            else:
                # Old format fallback
                x, y = lm
                
        # Draw landmarks grouped by facial feature
        for feature, indices in feature_groups.items():
            feature_color = feature_colors.get(feature, color)
            
            # Draw points
            for idx in indices:
                if idx in landmark_dict:
                    cv2.circle(frame, landmark_dict[idx], 1, feature_color, -1)
            
            # Connect adjacent points in the same feature
            for i in range(len(indices) - 1):
                idx1, idx2 = indices[i], indices[i + 1]
                if idx1 in landmark_dict and idx2 in landmark_dict:
                    cv2.line(frame, landmark_dict[idx1], landmark_dict[idx2], feature_color, 1)
                    
        # Add feature labels
        for feature, indices in feature_groups.items():
            # Find center of feature
            points = [landmark_dict[idx] for idx in indices if idx in landmark_dict]
            if points:
                center_x = sum(p[0] for p in points) // len(points)
                center_y = sum(p[1] for p in points) // len(points)
                # Add small label
                cv2.putText(frame, feature, (center_x, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, feature_colors[feature], 1)
    
    def calculate_fps(self):
        """Calculate frames per second"""
        current_time = time.time()
        fps = 1 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time
        self.fps_history.append(fps)
        return np.mean(self.fps_history)
    
    def draw_emotion_bars(self, frame, x, y, w, emotion_probs, bar_height=8, spacing=4):
        """Draw emotion probability bars with gradient colors"""
        bar_width = 100
        start_y = y
        
        # Sort emotions by probability for better visualization
        sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
        
        for i, (emotion, prob) in enumerate(sorted_emotions):
            # Skip if probability is very low
            if prob < 0.01:
                continue
                
            # Draw bar background
            bar_x = x
            bar_y = start_y + i * (bar_height + spacing)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            
            # Draw probability bar with color based on emotion
            prob_width = int(prob * bar_width)
            
            # Color mapping for emotions
            color_map = {
                'Happy': (0, 255, 0),     # Green for happy
                'Neutral': (0, 0, 0),     # Black for neutral
                'Sad': (255, 0, 0),       # Blue for sad (BGR format)
                'Angry': (0, 0, 255)      # Red for angry
            }
            
            # Use the corresponding color or default to white
            color = color_map.get(emotion, (255, 255, 255))
            
            # Use gradient for better visualization
            for j in range(prob_width):
                # Calculate gradient color
                alpha = j / bar_width
                gradient_color = tuple(int(c * (0.5 + 0.5 * alpha)) for c in color)
                
                # Draw vertical line with gradient color
                cv2.line(frame, (bar_x + j, bar_y), (bar_x + j, bar_y + bar_height), gradient_color, 1)
            
            # Draw emotion label with emoji
            label = f"{self.emotion_emojis.get(emotion, '')} {emotion}: {prob*100:.1f}%"
            cv2.putText(frame, label, (bar_x + bar_width + 10, bar_y + bar_height - 1),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def detect_emotions(self, frame):
        """Detect and visualize emotions in the given frame"""
        # Make a copy of the frame for visualization
        vis_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        # Record start time for performance monitoring
        start_time = time.time()
        
        # Show loading message if models aren't loaded yet
        if not self.models_loaded:
            # Check if loading thread is still alive
            if self.loading_thread and self.loading_thread.is_alive():
                cv2.putText(vis_frame, "Loading models, please wait...", (20, frame_height // 2),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                return vis_frame
            else:
                # If thread is done but models not loaded, show error
                cv2.putText(vis_frame, "Error loading models! Using fallback mode.", (20, frame_height // 2),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        try:
            # Detect faces using MediaPipe (primary method)
            faces = self.detect_faces_mediapipe(frame)
            
            # If no faces found, try cascade detector as fallback
            if len(faces) == 0:
                faces = self.detect_faces_cascade(frame)
            
            # Track faces across frames
            tracked_faces = self.track_faces(faces, frame_width, frame_height)
            
            # Process each tracked face
            for face_id, face_data in tracked_faces.items():
                x, y, w, h = face_data['rect']
                
                # Ensure face region is valid
                if x < 0 or y < 0 or x+w > frame_width or y+h > frame_height or w <= 0 or h <= 0:
                    continue
                    
                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size == 0:
                    continue
                
                # Detect landmarks
                landmarks = self.detect_landmarks(frame, (x, y, w, h, 0))
                
                # Predict emotion
                emotion, confidence, emotion_probs = self.predict_emotion(face_roi)
                
                # Apply temporal smoothing
                try:
                    smooth_emotion, smooth_confidence, smooth_probs = self.smooth_emotion(
                        face_id, emotion, confidence, emotion_probs
                    )
                except Exception as e:
                    # Fallback if smoothing fails
                    smooth_emotion, smooth_confidence, smooth_probs = emotion, confidence, emotion_probs
                
                # Draw face rectangle with color based on emotion
                emotion_color = self.emotion_colors.get(smooth_emotion, (255, 255, 255))
                
                # Draw rounded rectangle for better aesthetics
                cv2.ellipse(vis_frame, (x, y), (10, 10), 0, 90, -90, emotion_color, 2)
                cv2.ellipse(vis_frame, (x + w, y), (10, 10), 0, 180, 0, emotion_color, 2)
                cv2.ellipse(vis_frame, (x, y + h), (10, 10), 0, 0, 90, emotion_color, 2)
                cv2.ellipse(vis_frame, (x + w, y + h), (10, 10), 0, -90, 180, emotion_color, 2)
                
                cv2.line(vis_frame, (x + 10, y), (x + w - 10, y), emotion_color, 2)
                cv2.line(vis_frame, (x, y + 10), (x, y + h - 10), emotion_color, 2)
                cv2.line(vis_frame, (x + 10, y + h), (x + w - 10, y + h), emotion_color, 2)
                cv2.line(vis_frame, (x + w, y + 10), (x + w, y + h - 10), emotion_color, 2)
                
                # Draw facial landmarks if available
                if landmarks:
                    self.draw_landmarks(vis_frame, landmarks, emotion_color)
                
                # Create a larger emotion display box at the top of the frame
                emotion_box_height = 60
                emotion_box_width = 300
                emotion_box_x = (frame_width - emotion_box_width) // 2
                emotion_box_y = 30
                
                # Draw semi-transparent background
                overlay = vis_frame.copy()
                cv2.rectangle(overlay, 
                             (emotion_box_x, emotion_box_y), 
                             (emotion_box_x + emotion_box_width, emotion_box_y + emotion_box_height), 
                             emotion_color, -1)
                cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
                
                # Add large emoji and emotion text
                emoji = self.emotion_emojis[smooth_emotion]
                emotion_text = f"{emoji} {smooth_emotion.upper()} ({smooth_confidence:.1f}%)"
                
                # Draw large emotion text
                cv2.putText(vis_frame, emotion_text, 
                           (emotion_box_x + 10, emotion_box_y + 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                # Add face ID for tracking visualization
                cv2.putText(vis_frame, f"ID: {face_id}", (x, y - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Calculate and display FPS
            fps = self.calculate_fps()
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (10, frame_height - 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(vis_frame, timestamp, (frame_width - 230, frame_height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add mode information
            if not self.models_loaded:
                cv2.putText(vis_frame, "FALLBACK MODE (Limited accuracy)", (10, frame_height - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Add instructions
            cv2.putText(vis_frame, "Enhanced Emotion Detection System", (10, frame_height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        except Exception as e:
            # If anything fails, show error message
            cv2.putText(vis_frame, f"Error: {str(e)}", (20, frame_height // 2),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(f"Error in detection: {e}")
            
        return vis_frame
    
    def run(self):
        """Run the emotion detection on webcam feed with advanced features"""
        cap = cv2.VideoCapture(0)
        
        # Try to set HD resolution for better results
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Create window with trackbars for adjustments
        cv2.namedWindow('Enhanced Emotion Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Enhanced Emotion Detection', 1280, 720)
        
        print("Starting enhanced emotion detection system...")
        print("Press 'q' to quit")
        print("Press 's' to save screenshot")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video. Check camera connection.")
                break
            
            # Process frame for emotion detection
            processed_frame = self.detect_emotions(frame)
            
            # Display the processed frame
            cv2.imshow('Enhanced Emotion Detection', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Quit on 'q' press
            if key == ord('q'):
                break
                
            # Save screenshot on 's' press
            elif key == ord('s'):
                filename = f"emotion_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Screenshot saved as {filename}")
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Release resources
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
        
        print("Emotion detection system stopped.")

class DeepFaceEmotionDetector:
    """Advanced emotion detection using deep learning models"""
    
    def __init__(self):
        # Initialize mediapipe face mesh for landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Create face detection and face mesh instances
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Simplified emotion labels and emojis - REMOVED FEAR as requested
        self.emotion_emojis = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'neutral': '😐',
        }
        
        # Color coding for emotions as requested
        self.emotion_colors = {
            'happy': (0, 255, 0),    # Green for happy
            'neutral': (0, 0, 0),    # Black for neutral
            'sad': (255, 0, 0),      # Blue for sad (Note: BGR format, so this is blue)
            'angry': (0, 0, 255),    # Red for angry
        }
        
        # Performance metrics
        self.last_frame_time = time.time()
        self.fps_history = []
        
        # Add temporal smoothing
        self.emotion_history = deque(maxlen=15)  # Increase history buffer for smoother transitions
        
        # Actions for first run
        print("DeepFace emotion recognition system initialized")
    
    def smooth_emotion(self, emotion, confidence):
        """Apply temporal smoothing to reduce fluctuations"""
        # Add current prediction to history
        self.emotion_history.append((emotion, confidence))
        
        # Not enough history yet
        if len(self.emotion_history) < 3:
            return emotion, confidence
            
        # Count occurrences of each emotion in history
        emotion_counts = {}
        emotion_confidences = {}
        
        # Apply recency weighting (more recent predictions have higher weight)
        weights = [0.5, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        weights = weights[-len(self.emotion_history):]
        
        for i, (e, c) in enumerate(self.emotion_history):
            weight = weights[i]
            if e not in emotion_counts:
                emotion_counts[e] = 0
                emotion_confidences[e] = 0
            
            emotion_counts[e] += weight
            emotion_confidences[e] += c * weight
            
        # Get most frequent emotion
        if emotion_counts:
            smoothed_emotion = max(emotion_counts, key=emotion_counts.get)
            avg_confidence = emotion_confidences[smoothed_emotion] / emotion_counts[smoothed_emotion]
            return smoothed_emotion, avg_confidence
            
        return emotion, confidence
        
    def detect_emotions(self, frame):
        """Detect emotions using DeepFace pre-trained model"""
        # Make a copy of the frame for visualization
        vis_frame = frame.copy()
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        
        # Display FPS
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        cv2.putText(vis_frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        try:
            # Detect faces using MediaPipe for better speed
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_detection_results = self.face_detection.process(rgb_frame)
            
            if face_detection_results.detections:
                # Get face bounding box
                for detection in face_detection_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x = max(0, int(bboxC.xmin * iw))
                    y = max(0, int(bboxC.ymin * ih))
                    w = min(int(bboxC.width * iw), iw - x)
                    h = min(int(bboxC.height * ih), ih - y)
                    
                    # Get landmarks
                    mesh_results = self.face_mesh.process(rgb_frame)
                    
                    # Use DeepFace for emotion detection (pre-trained model)
                    try:
                        face_roi = frame[y:y+h, x:x+w]
                        if face_roi.size > 0:
                            # Use DeepFace pre-trained model for emotion detection
                            analysis = DeepFace.analyze(
                                face_roi, 
                                actions=['emotion'],
                                enforce_detection=False,
                                detector_backend='opencv'
                            )
                            
                            # Get dominant emotion and confidence
                            emotions = analysis[0]['emotion']
                            
                            # Filter out 'fear' emotion as requested
                            if 'fear' in emotions:
                                del emotions['fear']
                            if 'surprise' in emotions:
                                del emotions['surprise']
                            if 'disgust' in emotions:
                                del emotions['disgust']
                                
                            # Get dominant emotion after filtering
                            dominant_emotion = max(emotions, key=emotions.get)
                            confidence = emotions[dominant_emotion]
                            
                            # Apply temporal smoothing to reduce fluctuations
                            smooth_emotion, smooth_confidence = self.smooth_emotion(dominant_emotion, confidence)
                            
                            # Get color for the emotion
                            emotion_color = self.emotion_colors.get(smooth_emotion.lower(), (255, 255, 255))
                            
                            # Display emotion with emoji
                            emoji = self.emotion_emojis.get(smooth_emotion.lower(), '')
                            emotion_text = f"{emoji} {smooth_emotion.upper()} ({smooth_confidence:.1f}%)"
                            
                            # Draw face rectangle with emotion color
                            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), emotion_color, 2)
                            
                            # Draw emotion text
                            cv2.putText(vis_frame, emotion_text, 
                                       (x, y - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                                       emotion_color, 2)
                            
                            # Draw facial landmarks if available
                            if mesh_results.multi_face_landmarks:
                                for face_landmarks in mesh_results.multi_face_landmarks:
                                    self.mp_drawing.draw_landmarks(
                                        vis_frame,
                                        face_landmarks,
                                        self.mp_face_mesh.FACEMESH_CONTOURS,
                                        landmark_drawing_spec=None,
                                        connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                            color=emotion_color, thickness=1, circle_radius=1)
                                    )
                    except Exception as e:
                        # If DeepFace fails, show error
                        cv2.putText(vis_frame, f"Error: {str(e)[:30]}", 
                                   (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                                   (0, 0, 255), 2)
            else:
                # No face detected
                cv2.putText(vis_frame, "No face detected", (30, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
        except Exception as e:
            # Handle general errors
            cv2.putText(vis_frame, f"Error: {str(e)[:30]}", (30, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            print(f"Error in emotion detection: {e}")
            
        return vis_frame
    
    def run(self):
        """Run emotion detection on webcam feed"""
        cap = cv2.VideoCapture(0)
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("DeepFace emotion detection started. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video. Check camera connection.")
                break
            
            # Flip the frame horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Process frame for emotion detection
            processed_frame = self.detect_emotions(frame)
            
            # Display the processed frame
            cv2.imshow('DeepFace Emotion Detection', processed_frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Emotion detection stopped.")

def check_requirements():
    """Check and install missing requirements"""
    required_packages = {
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'requests': 'requests',
        'tqdm': 'tqdm',
        'mediapipe': 'mediapipe'
    }
    
    missing_packages = []
    
    for package, module in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages. Please install:")
        for package in missing_packages:
            print(f"  pip install {package}")
        return False
    
    return True

def fallback_emotion_detection():
    """Run a simple fallback emotion detection using only OpenCV"""
    print("Starting fallback emotion detection (basic features only)...")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Error: Failed to load face detector.")
        cap.release()
        return
        
    # Create window
    cv2.namedWindow('Fallback Emotion Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Fallback Emotion Detection', 1280, 720)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture video. Check camera connection.")
            break
            
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Process each face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Simple text
            cv2.putText(frame, "Face Detected", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add fallback mode label
        cv2.putText(frame, "FALLBACK MODE (No emotion detection)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Fallback Emotion Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Fallback system stopped.")

def main():
    """Main function to run the enhanced emotion detection system"""
    try:
        # Check requirements
        if not check_requirements():
            return
            
        try:
            # Create and run emotion detector
            detector = EnhancedEmotionDetector()
            detector.run()
        except Exception as e:
            print(f"Error in main detection system: {e}")
            print("Trying fallback mode...")
            fallback_emotion_detection()
        
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Make sure your camera is connected and working")
        print("2. Ensure you have sufficient permissions to access the camera")
        print("3. Install required packages: pip install opencv-python numpy requests tqdm mediapipe")
        print("4. Try running with administrator/root privileges")
        
        # Final fallback - just try to show webcam feed
        try:
            print("Attempting to show basic webcam feed...")
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                for _ in range(10):  # Try for a few frames
                    ret, frame = cap.read()
                    if ret:
                        cv2.imshow('Basic Webcam Test', frame)
                        cv2.waitKey(100)
                cv2.destroyAllWindows()
                cap.release()
                print("Webcam test completed.")
            else:
                print("Could not access webcam.")
        except:
            print("Failed to access webcam. Please check your hardware.")

if __name__ == "__main__":
    main()