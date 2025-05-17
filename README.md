
🧠 Image Classification and Object Detection Using PASCAL VOC 2007

Subject: Machine Vision  
Dataset: PASCAL VOC 2007  
---

📌 Project Overview

This project implements core computer vision tasks including:

- Image Classification
- Object Detection
- Object Recognition

Using the PASCAL VOC 2007 dataset, we explore image processing, feature extraction, and model evaluation techniques.

---

🗂 Dataset Structure and Preparation

We used the PASCAL VOC 2007 dataset with the following structure:

VOCdevkit/
  VOC2007/
    Annotations/          # XML files with bounding boxes and labels
    JPEGImages/           # Raw images
    ImageSets/
      Main/               # Training/testing split
    SegmentationClass/    # Segmentation masks
    SegmentationObject/   # Object segmentation

- Training set size: ~5,000 images  
- Test set size: ~5,000 images  
- Annotation format: Pascal VOC XML  

---

🧹 Image Preprocessing

We applied the following preprocessing steps:

- Resize images to 224x224
- Convert to grayscale (for HOG/SIFT)
- Tensor transformation (for neural models)
- Bounding box visualization from annotation files (using XML parser)

---

🧬 Feature Extraction Techniques

1. HOG (Histogram of Oriented Gradients)
- Implemented with skimage.feature.hog
- Captures edge orientations and shape information
- Works on 224x224 grayscale images

2. SIFT (Scale-Invariant Feature Transform)
- Implemented using OpenCV
- Extracts keypoints and descriptors
- Visualized using OpenCV draw functions
- Converted to fixed-size histograms using Bag of Visual Words (BoVW)

---

🧠 Image Classification

Model:
- Support Vector Machine (SVM) with linear kernel

Input:
- HOG or SIFT + BoVW features

Output:
- Object class (e.g., car, dog, person)

Evaluation:
- Train/Test Split: 80/20
- Metrics: Accuracy, Confusion Matrix
- Visualization: True vs predicted labels with green/red highlights

Results:
- SVM performed well on many categories, though overlapping features caused some misclassifications

---

🎯 Object Detection and Recognition

Model:
- Faster R-CNN with ResNet-50 + FPN backbone  
- Pretrained on COCO, fine-tuned on PASCAL VOC

Inference:
- Images preprocessed using PyTorch transforms
- Model outputs: bounding boxes, class labels, confidence scores
- Threshold: 0.5 for predictions

Visualization:
- Ground Truth (Red) vs Predicted Boxes (Green)
- Labels and confidence scores shown for each prediction

Results:
- Accurate detection in complex scenes
- Pretrained model reduced training time significantly

---

📊 Visualizations & Analysis

- Comparison of HOG vs SIFT performance
- Visualization of classification results
- Bounding box detection accuracy visualized
- SIFT more robust; HOG faster and simpler

---

✅ Conclusion

- HOG: Fast and simple, but struggles with complex patterns  
- SIFT: More robust, but needs BoVW to standardize input  
- Faster R-CNN: Excellent detection performance using transfer learning  

This project showcases how traditional and deep learning-based techniques can be applied to real-world computer vision problems using a standardized dataset.

---

📘 References

- PASCAL VOC Dataset: http://host.robots.ox.ac.uk/pascal/VOC/
- Faster R-CNN Paper: https://arxiv.org/abs/1506.01497
- Libraries: OpenCV, PyTorch, scikit-learn, skimage
