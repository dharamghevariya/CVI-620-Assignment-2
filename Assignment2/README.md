# CVI620 - Assignment 2: Image Classification

**Dharam Mehulbhai Ghevariya**  
Student ID: 136270220  
Email: dmghevariya@myseneca.ca  

Seneca Polytechnic | Fall 2025

---

## Overview

This assignment explores image classification using traditional machine learning methods. It consists of two questions, each focusing on a different classification problem.

---

## Questions

### Question 1: Cat vs Dog Classification

Classify images of cats and dogs using the provided dataset of 2000 training images. The goal is to compare different algorithms and tune parameters to achieve the best possible results.

**Methods Used:** K-Nearest Neighbors (KNN), Logistic Regression

**Best Result:** KNN with K=17 achieved 70% accuracy on the test set

| Document | Description |
|----------|-------------|
| [Q1/instructions.md](Q1/instructions.md) | How to run the scripts, requirements, and usage guide |
| [Q1/answer.md](Q1/answer.md) | Complete analysis, parameter tuning results, and conclusions |

**Key Files:**
- `Q1/cat_dog_classification.py` - Main training script
- `Q1/predict_image.py` - Predict on a single image
- `Q1/test_internet_images.py` - Test on internet images

---

### Question 2: MNIST Digit Classification

Classify handwritten digits (0-9) from the MNIST dataset containing 60,000 training images. The goal is to achieve at least 90% accuracy.

**Methods Used:** K-Nearest Neighbors (KNN), Logistic Regression

**Best Result:** KNN with K=3 achieved 97.05% accuracy on the test set

| Document | Description |
|----------|-------------|
| [Q2/instructions.md](Q2/instructions.md) | How to run the scripts, requirements, and usage guide |
| [Q2/answer.md](Q2/answer.md) | Complete analysis, parameter tuning results, and conclusions |

**Key Files:**
- `Q2/mnist_classification.py` - Main training script
- `Q2/predict_digit.py` - Predict on custom digit images

---

## Requirements

Install the required Python packages:

```bash
pip install numpy opencv-python scikit-learn matplotlib joblib pandas
```

---

## Quick Start

### Training Models

```bash
# Train Cat vs Dog classifier
cd Q1
python cat_dog_classification.py

# Train MNIST digit classifier
cd Q2
python mnist_classification.py
```

### Making Predictions

```bash
# Predict cat or dog
cd Q1
python predict_image.py path/to/image.jpg

# Predict digit
cd Q2
python predict_digit.py path/to/digit.png
```

---

## Results Summary

| Question | Dataset | Best Method | Accuracy | Target |
|----------|---------|-------------|----------|--------|
| Q1 | Cat vs Dog | KNN (K=17) | 70% | - |
| Q2 | MNIST | KNN (K=3) | 97.05% | 90% ✓ |

---

## Repository Structure

```
Assignment2/
├── README.md                    # This file
├── .gitignore                   # Excludes large model files
├── Q1/                          # Cat vs Dog Classification
│   ├── instructions.md          # Usage guide
│   ├── answer.md                # Analysis and results
│   ├── cat_dog_classification.py
│   ├── predict_image.py
│   ├── test_internet_images.py
│   ├── train/                   # Training images
│   ├── test/                    # Test images
│   ├── images/                  # Internet test images
│   └── saved_models/            # Trained models
└── Q2/                          # MNIST Digit Classification
    ├── instructions.md          # Usage guide
    ├── answer.md                # Analysis and results
    ├── mnist_classification.py
    ├── predict_digit.py
    ├── mnist_train.csv          # Training data
    ├── mnist_test.csv           # Test data
    ├── images/                  # Internet test images
    └── saved_models/            # Trained models
```

---

## Notes

- Trained model files (`.pkl`) are not included in the repository due to size limits. Run the training scripts to generate them.
- The `mnist_train.csv` file is also excluded due to its large size (104MB).
