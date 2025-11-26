# Cat vs Dog Image Classification - Instructions

## Overview

This project classifies images of cats and dogs using two machine learning algorithms: K-Nearest Neighbors (KNN) and Logistic Regression. KNN works by finding the K closest training images to a new image and taking a majority vote, while Logistic Regression finds a linear decision boundary that best separates the two classes. Both models are trained on 2000 images (1000 cats and 1000 dogs) and evaluated on a test set.

For the complete analysis and results, see `answer.md`.

---

## Folder Structure

```
Q1/
├── cat_dog_classification.py    # Main training script
├── predict_image.py             # Prediction script for single images
├── test_internet_images.py      # Script to test all internet images
├── instructions.md              # This file (how to use)
├── answer.md                    # Complete answer with analysis
├── classification_results.png   # Visualization of results
├── saved_models/                # Trained models (scaler.pkl, knn_model.pkl, logistic_regression_model.pkl)
├── images/                      # Internet images for testing
├── train/                       # Training dataset (Cat/ and Dog/ folders)
└── test/                        # Test dataset (Cat/ and Dog/ folders)
```

---

## Requirements

Before running the scripts, install the required Python packages:

```bash
pip install numpy opencv-python scikit-learn matplotlib joblib
```

---

## How to Use

### Training the Models

To train both models, run the main classification script:

```bash
python cat_dog_classification.py
```

This script loads and preprocesses all training images, trains KNN with different K values (1 through 21) to find the best K, trains Logistic Regression with different C values (0.001 through 100) to find the best C, evaluates both models on the test set, saves the trained models to the `saved_models/` folder, and generates a comparison visualization saved as `classification_results.png`.

### Testing on Internet Images

To evaluate the saved models on all images in the `images/` folder, run:

```bash
python test_internet_images.py
```

### Predicting on a Single Image

After training, you can predict on any new image using:

```bash
python predict_image.py path/to/your/image.jpg
```

For example: `python predict_image.py images/cat1.jpg` or `python predict_image.py images/dog1.jpg`

---

## Algorithm Details

**K-Nearest Neighbors (KNN)** finds the K closest training images to a new image and classifies it based on majority vote. The parameter K (number of neighbors) was tuned by testing values from 1 to 21, with K=17 giving the best results. Odd values are used to avoid ties in binary classification.

**Logistic Regression** finds a linear decision boundary that best separates cats from dogs. The regularization parameter C controls model complexity - smaller C means stronger regularization (simpler model), while larger C means weaker regularization (more complex model). C=0.001 worked best for this dataset.

---

## Preprocessing

All images go through the same preprocessing pipeline: they are loaded using OpenCV, resized to 64x64 pixels for consistency, converted to grayscale to reduce the number of features, flattened from a 2D image to a 1D array (resulting in 4096 features), and then normalized using StandardScaler so that pixel values have zero mean and unit variance.

---

## Quick Results

| Model               | Test Accuracy | Internet Images Accuracy |
|---------------------|---------------|--------------------------|
| KNN (K=17)          | 70%           | 55.6% (5/9)              |
| Logistic Regression | 60%           | 77.8% (7/9)              |

For detailed results and analysis, see `answer.md`.

---

**Dharam Mehulbhai Ghevariya**  
Email: dmghevariya@myseneca.ca  
Student ID: 136270220  

CVI620 - Computer Vision | Seneca Polytechnic | Fall 2025
