# CIFAR-10 Object Recognition using ResNet50 with Transfer Learning

This repository contains a notebook for training a Convolutional Neural Network (CNN) to recognize objects in the CIFAR-10 dataset using the ResNet50 architecture with transfer learning.


## Introduction

This project aims to classify images from the CIFAR-10 dataset into one of ten categories using deep learning techniques. The model leverages transfer learning by utilizing the ResNet50 architecture pre-trained on ImageNet, fine-tuning it for the CIFAR-10 classification task.

## Dataset

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The dataset can be downloaded using the Kaggle API: kaggle competitions download -c cifar-10

## Data Preprocessing
## Labels Processing

The labels are processed to make them suitable for the classification task.

## Image Processing

Images are processed through resizing, normalization, and augmentation to enhance the model's ability to generalize. The processing steps include:

- Resizing: Images are resized to match the input size expected by the ResNet50 model.
- Normalization: Pixel values are scaled to a range suitable for the model.

## Train Test Split

The dataset is split into training and testing sets

## Model Architecture

### Building the Neural Network

The neural network is built using the TensorFlow/Keras framework. The model starts with the ResNet50 architecture, excluding the top layers, and adds custom layers to adapt it to the CIFAR-10 classification task.

### ResNet50

ResNet50 is a 50-layer deep convolutional network that has been pre-trained on the ImageNet dataset. It serves as the feature extractor in this project, with its top layers fine-tuned on the CIFAR-10 dataset.

## Training

The model is trained using a combination of transfer learning and fine-tuning techniques. The training process includes:

- **Optimizer**: Adam optimizer is used with an appropriate learning rate.
- **Loss Function**: Categorical Crossentropy is used as the loss function.
- **Metrics**: Accuracy is tracked as the primary metric during training.

## Evaluation

After training, the model is evaluated on the test set. The evaluation metrics include:

- **Accuracy**: The percentage of correctly classified images.

## Results

The trained model achieves satisfactory accuracy on the CIFAR-10 dataset, demonstrating the effectiveness of transfer learning using ResNet50.

## Conclusion

This project successfully demonstrates the application of transfer learning to the CIFAR-10 dataset using the ResNet50 architecture. Future improvements could include hyperparameter tuning and exploring other architectures for further performance gains.
