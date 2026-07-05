# Smart Vision System for Footpath Hazard & Obstruction Detection

## Overview

Smart Vision System for Footpath Hazard & Obstruction Detection is a deep learning–based computer vision project designed to improve pedestrian safety by automatically identifying footpath regions and detecting hazards present within them.

Unlike conventional object detection systems that analyze the entire image, this project follows a **two-stage detection pipeline**. The first stage identifies the walkable footpath region, while the second stage performs object detection only inside the detected footpath. This approach minimizes unnecessary detections outside the pedestrian area and improves the relevance of detected hazards.

The proposed system can be applied in smart city infrastructure monitoring, municipal maintenance, pedestrian safety analysis, and urban accessibility assessment.

---

# Project Objectives

- Detect the walkable footpath region from street images.
- Identify hazards and obstructions present on the detected footpath.
- Reduce false detections by restricting object detection to pedestrian regions.
- Develop a lightweight and efficient AI-based monitoring system.
- Provide a user-friendly interface for real-time analysis.

---

# Two-Stage Detection Pipeline

## Stage 1 – Footpath Detection

The first stage detects the complete footpath region from the input image.

This stage is responsible for:

- Identifying the walkable area
- Generating the Region of Interest (ROI)
- Eliminating unnecessary background regions

Output:

- Detected footpath region

---

## Stage 2 – Obstruction Detection

The detected footpath region is passed to the second model.

The second model detects hazards present only inside the footpath region.

Detected classes include:

- Light Pole
- Pothole / Ditch
- Roadside Stall
- Vehicle

By limiting detection to the footpath, the model focuses only on pedestrian-related hazards.

---

# System Workflow

Input Image
      ↓
Footpath Detection Model
      ↓
Extract Footpath Region (ROI)
      ↓
Hazard Detection Model
      ↓
Bounding Boxes + Class Labels + Confidence Scores
      ↓
Final Output

---

# Dataset

The project uses a custom annotated dataset collected from real-world urban environments.

### Stage 1 Dataset

Purpose:

- Footpath Detection

Annotation Type:

- Polygon Masks
- Footpath Region

### Stage 2 Dataset

Purpose:

- Hazard Detection

Annotation Format:

- YOLO Bounding Boxes

Classes:

| Class ID | Class Name |
|-----------|------------|
| 0 | Light Pole |
| 1 | Pothole / Ditch |
| 2 | Roadside Stall |
| 3 | Vehicle |

Dataset Split

- Training Images: 1900
- Validation Images: 250
- Test Images: Available

---

# Technologies Used

- Python
- Ultralytics YOLO
- OpenCV
- PyTorch
- Streamlit
- NumPy
- Pandas

---

# Features

- Two-stage AI pipeline
- Footpath region extraction
- Footpath hazard detection
- Multiple obstruction classes
- Image detection
- Video detection
- Real-time inference
- Streamlit web interface
- Lightweight deployment

---

# Model Performance

## Stage 1 Model

Performance depends on the trained footpath detection model.

## Stage 2 Hazard Detection Model

Final validation metrics obtained during training:

| Metric | Value |
|---------|--------|
| Precision | **60.72%** |
| Recall | **50.21%** |
| mAP@50 | **49.56%** |
| mAP@50-95 | **21.84%** |

> Results obtained after **100 training epochs** using the custom hazard detection dataset.

---

# Project Structure

```
Footpath-Hazard-Detection/
├── datasets/
│   ├── Footpath Dataset
│   └── Hazard Dataset
├── models/
├── weights/
├── app.py
├── train.py
├── detect.py
├── requirements.txt
└── README.md
```

---

# Future Improvements

- Improve footpath segmentation accuracy.
- Increase dataset diversity.
- Add more hazard categories.
- GPS-based hazard mapping.
- Municipal reporting dashboard.
- Mobile application integration.
- Real-time CCTV monitoring.

---

# Applications

- Smart City Monitoring
- Municipal Infrastructure Management
- Pedestrian Safety Assessment
- Urban Accessibility Analysis
- Automated Footpath Inspection
- Road Maintenance Support

---

# Author

**Ibadur Laskar**

M.Sc. Computer Science

Assam University
