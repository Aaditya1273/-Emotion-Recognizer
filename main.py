import cv2
import numpy as np
import time
from emotion_detection import DeepFaceEmotionDetector, EnhancedEmotionDetector, fallback_emotion_detection

def main():
    try:
        # Try to use DeepFaceEmotionDetector first
        print("Initializing DeepFace emotion detector...")
        emotion_detector = DeepFaceEmotionDetector()
        emotion_detector.run()
    except Exception as e:
        print(f"DeepFace detector failed: {e}")
        try:
            # Try EnhancedEmotionDetector as fallback
            print("Falling back to EnhancedEmotionDetector...")
            detector = EnhancedEmotionDetector()
            detector.run()
        except Exception as e2:
            print(f"Enhanced detector failed: {e2}")
            # Use basic fallback as last resort
            print("Using basic fallback detection...")
            fallback_emotion_detection()

if __name__ == "__main__":
    main() 