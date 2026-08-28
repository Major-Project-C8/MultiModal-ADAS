# MultiModal-ADAS

# Module 3: Dark Channel Prior Dehazing & YOLOv8 Road Damage Detection

## Overview
Module 3 forms the core computer vision engine for the Multi-Condition ADAS pipeline. Standard perception models suffer high false-negative rates in foggy or hazy conditions due to contrast degradation and light scattering. 

This module resolves visibility limitations by combining **Dark Channel Prior (DCP)** atmospheric dehazing with a **YOLOv8 deep learning detector** fine-tuned on the multi-national **RDD2022 dataset**. The pipeline restores foggy video frames in real time and classifies physical road surface anomalies into bounding box coordinates.

## Features
* **Atmospheric Dehazing Engine:** Implements single-image haze removal via Dark Channel Prior (DCP) to extract global atmospheric light and compute transmission maps.
* **Real-Time Object Detection:** Fine-tuned YOLOv8 (Nano/Small) architecture optimized for high-speed inference on localized road hazards.
* **Multi-Class Defect Categorization:** Detects and classifies 4 structural road damage types (Longitudinal Cracks, Transverse Cracks, Alligator Cracks, Potholes).
* **Pipeline Integration Ready:** Converts bounding box predictions into spatial Region-of-Interest (ROI) masks to feed down-stream stereo depth calculations (Module 4).

## Repository Structure
```text
├── dataset/
│   ├── data.yaml              # Dataset paths and class mapping configuration
│   ├── train/                 # 70% split (Images & YOLO .txt labels)
│   ├── val/                   # 20% split (Images & YOLO .txt labels)
│   └── test/                  # 10% split (Unseen evaluation frames)
├── models/
│   ├── best.pt                # Fine-tuned YOLOv8 model weights
│   └── dehaze.py              # DCP dehazing mathematical functions
├── notebooks/
│   └── Module3_Training.ipynb # Google Colab training & evaluation pipeline
├── outputs/                   # Sample predictions, confusion matrices, loss graphs
├── main_module3.py            # Integrated Dehazing -> Detection script
├── requirements.txt           # Python dependencies
└── README.md                  # Module documentation
