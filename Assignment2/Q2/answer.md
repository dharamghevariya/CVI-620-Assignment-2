# Question 2 - MNIST Digit Classification

## Overview

For this question, I worked with the MNIST dataset which contains 60,000 training images and 10,000 test images of handwritten digits from 0 to 9. Each image is 28x28 pixels, represented as a flattened vector of 784 values in the CSV file. My goal was to classify these digits using different machine learning methods and achieve at least 90% accuracy.

## Preprocessing

Before training, I normalized the pixel values by dividing by 255 to scale them from 0-255 to 0-1 range. This helps the algorithms converge faster and often improves accuracy since features are on a similar scale.

## Methods Used

I implemented two classification methods and tuned their parameters to find the best configuration for each.

### 1. K-Nearest Neighbors (KNN)

KNN classifies a digit by looking at the K closest training examples and taking a majority vote. I tested different values of K:

| K Value | Accuracy |
|---------|----------|
| 1       | 96.91%   |
| 3       | 97.05%   |
| 5       | 96.88%   |
| 7       | 96.94%   |

K=3 performed best with 97.05% accuracy. This makes sense because using a few neighbors helps reduce noise from individual outliers while still being specific enough to capture the digit patterns.

### 2. Logistic Regression

Logistic Regression learns a linear decision boundary for each class. The C parameter controls regularization strength, where smaller C means stronger regularization. I tested different values:

| C Value | Accuracy |
|---------|----------|
| 0.01    | 91.89%   |
| 0.1     | 92.58%   |
| 1.0     | 92.61%   |

C=1.0 gave the best result at 92.61%. Higher C means less regularization, allowing the model to fit the training data more closely. The accuracy is slightly lower than KNN because logistic regression assumes linear separability, which may not hold perfectly for complex digit patterns.

## Final Results

| Method              | Best Configuration | Accuracy | Target Met |
|---------------------|-------------------|----------|------------|
| KNN                 | K = 3             | 97.05%   | Yes        |
| Logistic Regression | C = 1.0           | 92.61%   | Yes        |

Both methods achieved the target of 90% accuracy.

## Best Method Analysis

KNN performed the best with 97.05% accuracy. Looking at the results, most digits are classified correctly with high precision and recall across all classes. The most common confusions happen between visually similar digits like 4 and 9, or 3 and 5, which is expected given how people write these numbers differently.

Logistic Regression achieved 92.61% accuracy, which still exceeds the 90% target. While it's simpler than KNN, it provides faster predictions since it doesn't need to compare against all training samples.

## Why These Methods Work Well on MNIST

MNIST is considered a relatively simple dataset for several reasons. The images are centered and normalized, the background is clean, and there's limited variation in style. This is why even basic methods like KNN can achieve excellent results. For more challenging handwriting datasets with varied styles and noise, deep learning methods like CNNs would likely be needed to maintain high accuracy.

## Saved Models

All trained models are saved in the `saved_models/` folder:
- `knn_mnist.pkl`
- `logistic_regression_mnist.pkl`
