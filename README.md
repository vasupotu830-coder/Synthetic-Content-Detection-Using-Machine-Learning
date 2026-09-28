# Synthetic Content Detection Using Machine Learning

A machine-learning-based web application for detecting potentially synthetic or manipulated visual content in images and videos using computer vision, handcrafted visual features, and a Random Forest classifier.

The system combines OpenCV-based preprocessing, YuNet face detection, feature extraction, machine learning classification, Flask web interfaces, and MySQL-based detection history.

---

## Project Overview

Synthetic media and deepfake content can be created by manipulating faces and other visual information in images and videos.

This project develops a machine-learning-based detection system that analyzes visual characteristics of uploaded media and classifies the content as:

- REAL
- FAKE

The current implementation supports:

- Image Detection
- Video Detection
- Detection History
- Dashboard
- Reports
- Analytics
- Model Information
- Application Settings
- MySQL Storage

The project is designed as a modular synthetic-content detection system, with Audio and Text detection planned as future extensions.

---

## Objectives

The main objectives of this project are:

1. Detect potentially manipulated visual content.
2. Extract meaningful visual characteristics from images.
3. Detect faces using OpenCV YuNet.
4. Train a machine-learning classifier using extracted features.
5. Support both image and video analysis.
6. Store detection results in a MySQL database.
7. Provide a web-based interface for users.
8. Provide analytical and historical information about previous detections.
9. Establish a baseline that can later be extended to more advanced synthetic-media detection methods.

---

# Features

## 1. Image Detection

Users can upload image files such as:

- JPG
- JPEG
- PNG

The system performs the following workflow:

1. Receives the uploaded image.
2. Preprocesses the image.
3. Detects faces using OpenCV YuNet.
4. Extracts handcrafted visual features.
5. Passes the features to the trained Random Forest classifier.
6. Predicts REAL or FAKE.
7. Calculates class probabilities.
8. Displays the detection result.
9. Stores the result in MySQL.

---

## 2. Video Detection

Users can upload video files such as:

- MP4
- AVI
- MOV
- MKV

The current video detection pipeline:

1. Loads the uploaded video.
2. Samples frames from the video.
3. Resizes the selected frames.
4. Extracts the same visual features used for image detection.
5. Performs frame-level classification.
6. Calculates REAL and FAKE probabilities for each sampled frame.
7. Aggregates the frame-level probabilities.
8. Produces a final video-level classification.
9. Stores the result in MySQL.

> The current video detector is a frame-sampling baseline. It does not model temporal relationships between consecutive frames.

---

## 3. Face Detection

The system uses the OpenCV YuNet face detector.

YuNet is used to:

- Locate faces in images.
- Determine whether a face is present.
- Calculate the detected face region.
- Calculate face-area-related features.
- Calculate face brightness and sharpness characteristics.

The model file used by the project is:

```text
models/face_detection_yunet_2026may.onnx