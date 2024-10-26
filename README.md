# 3D Hand Tracking Application

A Python application that uses hand tracking to manipulate a 3D cube in real-time using hand gestures.

## Features
- Real-time hand tracking using MediaPipe
- 3D cube manipulation with OpenGL
- Three interaction modes:
  - Position control (single hand pinch)
  - Rotation control (three finger pinch)
  - Scale control (two handed pinch)

## Requirements
- Python 3.12
- Webcam

## Installation
1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the application:
```bash
python main.py
```

### Controls
- Single hand index finger pinch: Move cube
- Three finger pinch: Rotate cube
- Two handed index finger pinch: Scale cube
- ESC: Exit application