# 😊 Real-Time Emotion Detection❤️

A powerful and intuitive real-time facial emotion detection system using **OpenCV** and **MediaPipe**. Detects emotions such as **Happy**, **Sad**, **Angry**, **Fear**, and **Neutral** from live webcam video feeds 🎥 — with smooth performance and high accuracy.

---

## ✨ Features:

✅ Real-time emotion detection from your webcam  
😀 Detects 4 core emotions: **Happy**, **Sad**, **Angry**, **Neutral**  
📊 Displays confidence/probability for each emotion  
😄 High accuracy for **smile/happy** detection  
🎛️ Clean UI with dynamic **confidence bars**  
⚡ Real-time **FPS (Frames Per Second)** counter for performance monitoring  

---

## 🛠 Requirements:

- Python 3.7 or above 🐍  
- Functional Webcam 🎥  
- Libraries listed in `requirements.txt`

---

## 🚀 Installation:

1. **Clone the repository** 📥
   ```bash
   git clone https://github.com/your-username/real-time-emotion-detector.git
   cd real-time-emotion-detector
   ```
2. **Install required packages** 📦
   ```bash
   pip install -r requirements.txt
   ```
---

## ▶️ How to Use:

1. **Run the application** 💻
   ```bash
   python main.py
   ```
2. Your webcam will activate and begin **real-time emotion detection** 🧠  
3. Press **'q'** to quit the application ❌
---

## 🧠 How It Works:

This system uses a hybrid approach combining facial geometry with fallback image analysis:

### 1. **Face Detection**  
🔍 Powered by **MediaPipe Face Mesh** to detect and track 468 facial landmarks in real-time.

### 2. **Facial Feature Analysis**  
Analyzes geometrical features like:
- 👄 **Mouth shape and curvature** (key for detecting smiles)
- 👁 **Eye aspect ratio**
- 🙄 **Eyebrow positioning**
- 😊 **Cheek area landmarks**

### 3. **Fallback Detection Logic**  
When landmark detection is weak, the system uses:
- 🖍 **Edge detection** in facial regions  
- 💡 **Brightness/intensity patterns**  
- 🔲 **Region-based texture analysis**
---

## ⚠️ Limitations:

- Works best in **good lighting** conditions 💡  
- Accuracy may vary depending on **webcam quality** 📸  
