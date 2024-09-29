Lung Cancer Detection Using CNN and VGG16
This repository contains the implementation of a deep learning model using Convolutional Neural Networks (CNN) and transfer learning with the VGG16 architecture to detect lung cancer. The project is based on the LUNA (LUng Nodule Analysis) dataset, which includes CT scan images of lungs.

Project Overview
Lung cancer is one of the leading causes of death worldwide, and early detection is crucial for improving survival rates. This project focuses on developing a robust model that can classify lung nodules as benign or malignant using deep learning techniques.

Key Components:
Convolutional Neural Networks (CNN): Used for feature extraction from the images.
VGG16 Transfer Learning: A pre-trained model on ImageNet is fine-tuned to suit lung cancer classification tasks.
LUNA Dataset: The LUng Nodule Analysis dataset provides labeled lung CT scan images.
Dataset
The dataset used for this project is LUNA16, which consists of thoracic CT scans with annotations for lung nodules. The images are preprocessed to extract lung nodule regions.

Dataset link: LUNA16 Dataset
Preprocessing:
The following steps are carried out to prepare the dataset for training:

Resizing: CT scan slices are resized to a standard input size for the VGG16 model (224x224 pixels).
Normalization: The pixel values are normalized to the range [0, 1].
Augmentation: Data augmentation techniques like rotation, flipping, and zooming are applied to increase the size of the dataset and reduce overfitting.
Model Architecture
1. CNN (Convolutional Neural Network)
A custom CNN architecture is designed to extract meaningful features from the lung nodule images. It consists of several convolutional layers, followed by pooling layers, and fully connected layers for final classification.
2. VGG16 Transfer Learning
VGG16 is a pre-trained network on ImageNet. For this project, the fully connected layers of VGG16 are fine-tuned, while the earlier layers are used for feature extraction. This allows leveraging learned features from natural images to medical images with a smaller dataset.
Architecture Highlights:
Input Layer: 224x224x3 (standard for VGG16)
Convolutional Layers: 13 convolutional layers (VGG16 backbone)
Fully Connected Layers: Custom layers for the final lung nodule classification task
Key Dependencies:
TensorFlow/Keras
NumPy
OpenCV
Scikit-learn
Matplotlib
Future Work
3D CNN: Extend the model to work on 3D CT scan volumes for more accurate predictions.
Larger Dataset: Incorporate additional datasets to improve the robustness of the model.
Web Application: Build a web-based interface where users can upload CT scans and get predictions in real-time.
License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
The LUNA16 dataset was used as the primary source of data for this project.
Thanks to the creators of VGG16 for the open-source pre-trained model.
