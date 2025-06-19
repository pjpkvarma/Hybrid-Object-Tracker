# 🛰️ Hybrid Drone Tracker: YOLOv8 + KCF with APCE-based Reinitialization

This project presents a robust hybrid visual tracking system for UAV (drone) videos by combining **YOLOv8 object detection** with **Kernelized Correlation Filter (KCF)** tracking, enhanced by the **APCE (Average Peak-to-Correlation Energy)** metric. The system alternates between detection and tracking based on the reliability of the correlation response, ensuring efficient and stable tracking across video frames.

## 📌 Overview

- **YOLO** is used for initial detection and for reinitialization when tracking quality degrades.
- **KCF** enables efficient frame-to-frame object tracking.
- **APCE** is computed from the correlation response map to evaluate the quality of the KCF tracker output.
- This approach is designed for **real-time UAV tracking** in videos where robustness and speed are both important.

## 📂 Repository Structure

```
Hybrid-Drone-Tracker/
├── main.py               # Main tracking script combining YOLO + KCF
├── kcf.py                # APCE-enhanced KCF tracker implementation
├── crazy_yolo.pt         # YOLOv8 model weights (to be added by user)
└── README.md             # Project documentation
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/hybrid-drone-tracker.git
cd hybrid-drone-tracker
```

### 2. Install dependencies

```bash
pip install opencv-python ultralytics numpy
```

### 3. Add YOLOv8 model

Place your trained YOLO weights as `crazy_yolo.pt` in the root directory.  
You can train a YOLO model using Ultralytics on your custom drone dataset.

### 4. Run the tracker

In `main.py`, update the video path to your drone footage:

```python
video_path = "path/to/your/video.mp4"
```

Then run:

```bash
python main.py
```

## 🧠 How It Works

- **Initialization**: YOLO detects the drone in the first frame.
- **Tracking**: KCF tracks the drone in subsequent frames.
- **Reinitialization**: If APCE falls below a threshold or KCF fails, YOLO is used again.
- **APCE**: Ensures only high-confidence KCF outputs are used, improving reliability.

## 🛠️ Customization

- Set `self.gray_feature = True` in `Tracker` class for grayscale tracking.
- Enable HOG visualization using `self.debug = True`.
- Tune tracking hyperparameters in `kcf.py`:  
  `sigma`, `padding`, `update_rate`, `max_patch_size`

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Jagadeswara Pavan Kumar Varma Pothuri**  
M.S. Robotics, University at Buffalo  
