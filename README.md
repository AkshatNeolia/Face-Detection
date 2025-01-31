# Face Recognition Attendance System

This is a **Face Recognition-based Attendance System** built using OpenCV, Face Recognition library, and Python. 
It captures faces in real-time using a webcam, compares them with pre-trained images, and marks attendance accordingly.

## Features
- Detects and recognizes faces in real-time.
- Marks attendance with timestamp in `Attendance.csv`.
- Displays a **green box** for recognized faces and a **red box** for unrecognized faces.
- Automatically stops if no recognized face appears for 10 seconds.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Face-Recognition-Attendance.git
cd Face-Recognition-Attendance
```

### 2. Create a Virtual Environment
```bash
conda create --name facedec python=3.9
conda activate facedec
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

1. Place training images in the `Training_images` folder. 
2. Run the face recognition script:
   ```bash
   python face_recognition_attendance.py
   ```
3. The system will recognize faces and mark attendance in `Attendance.csv`.

## Project Structure
```
📦 Face-Recognition-Attendance
 ┣ 📂 Training_images        # Folder containing face images
 ┣ 📜 face_recognition_attendance.py   # Main script
 ┣ 📜 requirements.txt       # Required dependencies
 ┣ 📜 Attendance.csv         # Logs attendance data
 ┗ 📜 README.md              # Project documentation
```

## Dependencies
- Python 3.9+
- OpenCV
- Face Recognition Library
- NumPy

## Acknowledgment
This project was developed using **Python and Conda virtual environment** named `facedec`.
