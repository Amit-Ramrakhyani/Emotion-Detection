# Emotion Detection using Facial images.

## Overview

This project aims to build a Convolutional Neural Network (CNN) model to detect emotions from facial images. The project includes data loading, model architecture design, training, evaluation, and prediction.

## Dataset

A pre-processed dataset containing facial images labeled with emotions (e.g., happy, sad, angry, neutral) is used.

## Reporsitory Structure

The repository is organized as follows:

```bash
emotion_detection/
├── data/
│   ├── train/
│   ├─- validation/
│   └── test/
├── app.py
├── emotion_detection_model.json
├── emotion_detection.weights.h5
├── model.ipynb
├── README.md
└── requirements.txt
```

## Dependencies

- Python 3.6+
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

## Getting Started

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Amit-Ramrakhyani/Emotion-Detection.git
```

2. Install the required packages:

```bash
python3 -m pip install -r requirements.txt
```

Alternatively, you can install the required packages manually:

```bash
pip install tensorflow keras opencv-python numpy matplotlib scikit-learn
```

3. To run the application, execute the following command:

```bash
python3 app.py
```

## Methodology

### 1. Data Loading

Data is loaded from the specified directory structure using the os and cv2 libraries. Images are read in grayscale mode to reduce complexity and computational load.

### 2. Data Preprocessing

- Label Encoding: Labels are converted to numerical values using LabelEncoder.
- One-Hot Encoding: The numerical labels are then converted to one-hot encoded vectors using to_categorical.
- Normalization: Image data is normalized by scaling pixel values to the range [0, 1].

### Data Augmentation

An `ImageDataGenerator` is used to create data generators for training, validation, and test sets. This helps in augmenting the dataset and improving the model's robustness.

### CNN Model Architecture

A Convolutional Neural Network (CNN) is designed with the following layers:

- **Convolutional Layers**: Extract features from the images.
- **MaxPooling Layers**: Reduce the spatial dimensions of the feature maps.
- **Dropout Layers**: Prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.
- **Flatten Layer**: Convert the 2D matrix data to a vector.
- **Dense Layers**: Perform classification based on the extracted features.
- **Output Layer**: Use softmax activation to output probabilities for each class.

  ### Model Architecture

  The CNN model consists of the following layers:

  ```bash
  ├── Conv2D: 32 filters, 3x3 kernel, ReLU activation
  ├── Conv2D: 64 filters, 3x3 kernel, ReLU activation
  ├── MaxPooling2D: 2x2 pool size
  ├── Dropout: 25%
  ├── Conv2D: 128 filters, 3x3 kernel, ReLU activation
  ├── MaxPooling2D: 2x2 pool size
  ├── Conv2D: 128 filters, 3x3 kernel, ReLU activation
  ├── MaxPooling2D: 2x2 pool size
  ├── Dropout: 25%
  ├── Flatten
  ├── Dense: 1024 units, ReLU activation
  ├── Dropout: 50%
  ├── Dense: 7 units, softmax activation
  ```

### 5. Compilation and Training

- **Optimizer**: Adam optimizer with an exponentially decaying learning rate is used.
- **Loss Function**: Categorical cross-entropy loss is used for multi-class classification.
- **Metrics**: Accuracy is used as the primary evaluation metric.

### 6. Model Evaluation

The model's performance is evaluated on the test set, and accuracy and loss curves are plotted.

### 7. Model Saving and Loading

The model architecture and weights are saved for future use. The saved model is then loaded for making predictions.

### 8. Real-Time Emotion Detection

A real-time emotion detection system is implemented using OpenCV to capture video from the webcam, detect faces, and predict emotions.

## Conclusion

The project successfully builds and evaluates a CNN model for emotion detection from facial images. The model architecture, training process, and evaluation results are well-documented. Real-time emotion detection is also implemented to demonstrate the practical applicability of the model.