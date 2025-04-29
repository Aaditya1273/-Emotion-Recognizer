# Real-time Emotion Detection

A real-time facial emotion detection system using OpenCV and MediaPipe. This application detects emotions such as Happy, Sad, Angry, Fear, and Neutral from webcam video feeds.

## Features

- Real-time emotion detection from webcam
- Detection of 5 emotions: Happy, Sad, Angry, Fear, Neutral 
- Probability display for each emotion
- High accuracy for smile/happy detection
- Clean and intuitive UI with emotion confidence bars
- FPS counter to monitor performance

## Requirements

- Python 3.7+
- Webcam
- Libraries listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script:

```bash
python main.py
```

2. The application will open your webcam and begin detecting emotions in real-time
3. Press 'q' to quit the application

## How It Works

The system uses a combination of techniques for emotion detection:

1. **Face Detection**: MediaPipe's face mesh is used to detect and track facial landmarks
2. **Facial Features Analysis**: Geometrical analysis of key facial features including:
   - Mouth shape and curvature (crucial for smile detection)
   - Eye aspect ratio
   - Eyebrow position
   - Cheek landmarks

3. **Fallback Detection**: When landmark detection isn't reliable, a more traditional image processing approach is used, analyzing:
   - Edge detection in different facial regions
   - Brightness patterns
   - Regional intensity differences

## Limitations

- Requires good lighting conditions for best results
- Performance may vary with different webcams
- Some emotions (like fear) may be harder to detect than others (like happiness)

## Customization

You can adjust detection thresholds in `emotion_detection.py` to fine-tune for your specific needs. 