# MNIST Digit Classification - Instructions

## Overview

This project classifies handwritten digits (0-9) from the MNIST dataset using two machine learning algorithms: K-Nearest Neighbors (KNN) and Logistic Regression. The dataset contains 60,000 training images and 10,000 test images, where each image is a 28x28 pixel grayscale image represented as a flattened vector of 784 values.

For the complete analysis and results, see `answer.md`.

---

## Requirements

Before running the script, install the required Python packages:

```bash
pip install numpy pandas scikit-learn joblib opencv-python matplotlib
```

---

## How to Use

### Training All Models

To train both classification models, run:

```bash
python mnist_classification.py
```

This script loads the MNIST training and test data from CSV files, normalizes pixel values to the 0-1 range, trains KNN with different K values (1, 3, 5, 7) to find the best K, trains Logistic Regression with different C values (0.01, 0.1, 1.0) to find the best regularization, evaluates both models on the test set, saves the best version of each model to the `saved_models/` folder, and prints a detailed comparison and classification report.

### Predicting on Custom Images

After training, you can predict digits from your own images using:

```bash
python predict_digit.py path/to/digit_image.png
```

You can also predict on multiple images at once:

```bash
python predict_digit.py image1.png image2.jpg image3.png
```

Or place images in the `images/` folder and run without arguments:

```bash
python predict_digit.py
```

**Tips for best results:**
- Use clear images of single handwritten digits
- White or light digits on dark background work best (MNIST format)
- Center the digit in the image
- The script auto-inverts if the background is light

---

## Algorithm Details

**K-Nearest Neighbors (KNN)** classifies a digit by finding the K closest training examples and taking a majority vote. K=3 performed best, achieving 97.05% accuracy. KNN works well on MNIST because similar digits tend to have similar pixel patterns.

**Logistic Regression** learns a linear decision boundary for each of the 10 digit classes. With C=1.0, it achieved 92.61% accuracy. While simpler than KNN, it provides faster predictions and still exceeds the 90% target.

---

## Preprocessing

The preprocessing is straightforward for MNIST since the images are already clean and centered. Each image is represented as 784 pixel values (28x28 flattened). The only preprocessing step is normalization, where pixel values are divided by 255 to scale them from 0-255 to 0-1 range. This helps algorithms converge faster and often improves accuracy.

---

## Quick Results

| Method              | Best Configuration | Accuracy | Target Met |
|---------------------|-------------------|----------|------------|
| KNN                 | K = 3             | 97.05%   | Yes        |
| Logistic Regression | C = 1.0           | 92.61%   | Yes        |

Both methods achieved the target of 90% accuracy. For detailed results and analysis, see `answer.md`.
