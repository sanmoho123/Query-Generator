# Real-Time Drowsiness Detection System 🚗💤  

## Overview  
This project implements a **real-time drowsiness and yawning detection system** using **deep learning and computer vision** to enhance driver safety. It detects signs of drowsiness based on **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)**, triggering alerts when necessary.  

## Features  
✅ **Real-time face and eye detection** using OpenCV and Dlib  
✅ **Deep learning-based drowsiness detection** with a trained CNN model  
✅ **Eye and mouth movement tracking** for fatigue detection  
✅ **Audio alert system** to warn drowsy drivers  
✅ **Optimized for real-time performance**  

## Technologies Used  
- **Python** 🐍  
- **OpenCV** (for image processing)  
- **Dlib** (for facial landmark detection)  
- **TensorFlow/Keras** (for deep learning)  
- **NumPy, SciPy, Imutils** (for numerical computations)  
- **Playsound** (for audio alerts)  

## How It Works  
1. **Face Detection**: Uses Dlib's pre-trained model to detect faces.  
2. **Eye & Mouth Landmark Detection**: Extracts facial landmarks from the detected face.  
3. **EAR & MAR Calculation**:  
   - EAR < 0.25 for 20+ frames → **Drowsiness Alert**  
   - MAR > 0.6 for 20+ frames → **Yawning Alert**  
4. **Alert System**: Plays an audio alert when a threshold is exceeded.  

## Installation  
### Prerequisites  
- Ensure Python is installed (`Python 3.7+ recommended`).  
- Download `shape_predictor_68_face_landmarks.dat` from [Dlib's official source](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the project directory.  

### Steps  
```bash
# Clone the repository
git clone https://github.com/yourusername/drowsiness-detection.git  
cd drowsiness-detection  

# Install dependencies
pip install -r requirements.txt  

# Run the script
python main.py  
