# MultiModal-ADAS

# Module 1 — Lane Detection & Solid/Dotted Boundary Classification

## Overview

Module 1 is responsible for lane-discipline detection in the MultiModal-ADAS system.

The module uses the TuSimple and CULane driving datasets and applies classical Computer Vision techniques, including Canny Edge Detection and the Probabilistic Hough Transform, to detect lane-boundary line segments from driving frames.

A geometric filtering and classification stage determines whether detected lane boundaries are Solid or Dotted based on segment length, continuity, and gap patterns.

The classification is smoothed across multiple frames to improve stability.

The lane-departure alert is triggered only when the vehicle crosses or approaches a solid lane boundary. Dotted-line crossings are ignored.

The module is intentionally implemented using classical Computer Vision rather than a deep-learning model, making it lightweight enough to operate alongside the other ADAS modules in an edge-processing pipeline.

---

## Features

- ROI-Based Preprocessing
  - Converts frames to grayscale
  - Applies Gaussian blur for noise reduction
  - Uses a trapezoidal Region of Interest (ROI)

- Canny Edge Detection
  - Detects lane-marking edges
  - Supports configurable low and high thresholds
  - Can be tuned for different lighting conditions

- Probabilistic Hough Transform
  - Uses `cv2.HoughLinesP`
  - Extracts candidate lane-boundary line segments
  - Filters non-lane lines using slope-based constraints

- Solid vs. Dotted Classification
  - Groups nearby and collinear segments
  - Uses segment length and gap patterns
  - Classifies lane boundaries as solid or dotted

- Temporal Smoothing
  - Uses a rolling-window majority vote
  - Reduces frame-to-frame classification flickering

- Lane Departure Alert
  - Estimates vehicle lateral position
  - Generates alerts for solid-line crossings or proximity
  - Ignores dotted-line crossings

- Pipeline Integration
  - Supports low-light information from Module 2
  - Can process dehazed frames from Module 3
  - Designed for integration with the complete MultiModal-ADAS pipeline

---

## System Pipeline

```
Input Driving Frame
        |
        v
Grayscale Conversion
        |
        v
Gaussian Blur
        |
        v
ROI Masking
        |
        v
Canny Edge Detection
        |
        v
Probabilistic Hough Transform
        |
        v
Slope & Geometric Filtering
        |
        v
Lane Boundary Grouping
        |
        v
Solid / Dotted Classification
        |
        v
Temporal Smoothing
        |
        v
Vehicle Lateral Offset Estimation
        |
        v
Solid-Line Departure Detection
        |
        v
Lane Departure Alert
```
## Module 2: Night-Time & Low-Light Processing

### Overview
Module 2 is responsible for detecting nighttime and low-light driving conditions in the MultiModal-ADAS system. The module uses the **BDD100K driving dataset** and analyzes driving images and video frames using the HSV (Hue, Saturation, Value) color space.

The Value (V) channel is used to estimate scene brightness, and threshold-based classification is applied to identify low-light conditions. The module provides lighting-condition information for integration with the other ADAS perception modules.

### Features
- **HSV-Based Brightness Analysis:** Converts driving images and video frames from BGR to HSV and extracts the V channel.
- **Low-Light Detection:** Uses V-channel brightness and thresholding to identify nighttime and low-light scenes.
- **Image & Video Processing:** Supports dataset images and frame-by-frame driving video processing.
- **Threshold Optimization:** Analyzes brightness values to determine a suitable low-light threshold.
- **Performance Evaluation:** Uses accuracy, precision, recall, F1-score, and confusion matrix.
- **Robustness Testing:** Evaluates performance under different nighttime and low-light conditions.
- **Pipeline Integration Ready:** Provides lighting-condition information for integration with the complete ADAS system.

### Topics Covered
**1. HSV & V-Channel Analysis**

- Converts driving images and video frames from BGR to HSV.
- Extracts the V channel and calculates brightness statistics.
- Uses V-channel brightness to characterize lighting conditions.

**2. Low-Light Classification**

- Applies a brightness threshold to classify normal-light and low-light conditions.
- Analyzes threshold values using selected driving samples.

**3. Image & Video Testing**

- Uses relevant **BDD100K driving data** for lighting-condition analysis.
- Processes video frames individually for low-light detection.
- Uses selected images and video clips for testing and demonstration.

**4. Performance Evaluation**

- Evaluates detection using accuracy, precision, recall, F1-score, and confusion matrix.
- Compares brightness values across different lighting conditions.

### Data & Testing
- **BDD100K Dataset:** Primary driving dataset used for testing and analyzing different lighting conditions.
- Selected dataset samples are used instead of storing the complete dataset in the repository.
- Both images and video frames are considered for evaluation.
- Additional nighttime and low-light driving clips may be used for robustness testing.

### Expected Outputs
- Normal-light/nighttime/low-light classification.
- Low-light alerts.
- Processed images and videos.
- V-channel brightness analysis.
- Threshold evaluation results.
- Classification metrics and confusion matrix.
- Lighting-condition information for integration with the complete ADAS pipeline.

### Repository Structure
- `text`

  - module2_night/

    - datasets/

      - images/
      - videos/
    - src/

      - night_detection.py
    - evaluation/

      - metrics.py
      - threshold_analysis.py
    - sample_data/

      - images/
      - videos/
      - results/
    - notebooks/

      - Module2_Night_Analysis.ipynb
    - config/

      - night_config.yaml
    - requirements.txt
    - README.md

### Dependencies
- OpenCV — Image/video processing and HSV conversion
- NumPy — Numerical and image-processing operations
- Pandas — Dataset and evaluation analysis
- Scikit-learn — Performance metrics
- Matplotlib — Brightness visualization

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Mean V-Channel Brightness

---

## Module 3: Dark Channel Prior Dehazing & YOLOv8 Road Damage Detection

### Overview
Module 3 forms the core computer vision engine for the Multi-Condition ADAS pipeline. Standard perception models suffer high false-negative rates in foggy or hazy conditions due to contrast degradation and light scattering. 

This module resolves visibility limitations by combining **Dark Channel Prior (DCP)** atmospheric dehazing with a **YOLOv8 deep learning detector** fine-tuned on the multi-national **RDD2022 dataset**. The pipeline restores foggy video frames in real time and classifies physical road surface anomalies into bounding box coordinates.

### Features
- **Atmospheric Dehazing Engine:** Implements single-image haze removal via Dark Channel Prior (DCP) to extract global atmospheric light and compute transmission maps.
- **Real-Time Object Detection:** Fine-tuned YOLOv8 (Nano/Small) architecture optimized for high-speed inference on localized road hazards.
- **Multi-Class Defect Categorization:** Detects and classifies 4 structural road damage types (Longitudinal Cracks, Transverse Cracks, Alligator Cracks, Potholes).
- **Pipeline Integration Ready:** Converts bounding box predictions into spatial Region-of-Interest (ROI) masks to feed down-stream stereo depth calculations (Module 4).

### Repository Structure
- `text`
  - datasets/
    - data.yaml
    - train/
    - val/
    - test/
  - models/
    - best.py
    - dehaze.py
    - notebooks/
    - Module3_Training.ipynb
    - Sample predictions, confusion matrices, loss graphs
  - requirements.txt
  - README.md

---

## Module 4: 3D Depth Estimation and System Integration

### Overview
The 3D Depth & Integration module is responsible for estimating the distance of detected road objects using stereo vision and integrating the outputs of the individual perception modules into a unified road-perception system. This module bridges the gap between 2D detection and 3D spatial awareness, enabling the ADAS system to not only detect hazards but also determine their exact location and distance from the vehicle.

### Features
- **Stereo Vision and Disparity Estimation:** Uses a stereo camera setup to capture the same road scene from different viewpoints and generates a disparity map using stereo-matching algorithms.
- **Depth Estimation:** Converts disparity to real-world distance using the formula `Z = (f × B) / d`, where `f` = focal length, `B` = baseline, and `d` = disparity.
- **KITTI Dataset Validation:** Validates the stereo-depth implementation using the KITTI Stereo dataset with metrics like MAE, RMSE, and depth error.
- **Pothole and Depth Integration:** Integrates with YOLO-based detection to combine bounding box, confidence, and estimated distance for each detected pothole.
- **System Integration:** Combines all perception modules (Lane Detection, Night/Low-Light Processing, YOLO Detection, Stereo Disparity, 3D Depth) into a unified running system.

### Topics Covered

**1. Stereo Vision and Disparity Estimation**
- The system uses a stereo camera setup consisting of a left and right camera. Both cameras capture the same road scene from slightly different viewpoints.
- The difference in the position of the same object between the left and right images is called disparity. A disparity map is generated by applying a stereo-matching algorithm to the two images.
- Higher disparity → object is closer
- Lower disparity → object is farther away

**2. Depth Estimation**
- After obtaining the disparity map, the distance of an object from the camera is calculated using `Z = (f × B) / d`.
- This converts the 2D stereo information into 3D depth information.

**3. Pothole and Depth Integration**
- The depth module is integrated with the YOLO-based pothole detection module.
- YOLO identifies: Pothole location, Bounding box, Detection confidence
- The depth map provides: Distance of the detected pothole from the camera
- Output: Pothole detected → Bounding box → Confidence → Estimated distance

**4. System Integration & Final Output**
- Combines all perception modules (Lane Detection, Night/Low-Light Processing, YOLO Detection, Stereo Disparity, 3D Depth) into a unified system.
- Final output provides: Lane information + Pothole detection + Pothole location + Pothole distance/depth

### Repository Structure
- `text`
  - stereo/
    - calibration/
      - calibration.py
      - stereo_calib.yaml
    - disparity/
      - sgbm.py
      - postprocess.py
    - depth/
      - depth_computer.py
    - fusion/
      - object_depth_associator.py
      - output_formatter.py
  - notebooks/
    - Module4_Stereo_Demo.ipynb
    - Calibration_Visualization.ipynb
  - evaluation/
    - metrics.py
  - sample_data/
    - left_frames/
    - right_frames/
    - depth_outputs/
  - config/
    - stereo_config.yaml
  - requirements.txt
  - README.md

### Dependencies
- OpenCV — stereo matching and image processing
- NumPy — mathematical computations
- PyTorch/TensorFlow — YOLO integration

### Expected Outputs
- Disparity maps and metric depth maps
- Annotated frames with bounding boxes and distance labels
- Combined road-perception output (lane info + pothole detection + depth)
