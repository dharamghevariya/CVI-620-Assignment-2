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

## Testing on Internet Images

I downloaded 18 handwritten digit images from the internet and tested the trained models on them. The images had various styles, backgrounds, and quality levels.

| Image | Actual Digit | KNN Prediction | Logistic Regression | Correct |
|-------|--------------|----------------|---------------------|---------|
| 0.png | 0 | 0 | 2 | KNN ✓ |
| 2(1).jpg | 2 | 2 | 2 | Both ✓ |
| 2.jpg | 2 | 1 | 5 | Both ✗ |
| 3.jpeg | 3 | 3 | 3 | Both ✓ |
| 4(2).png | 4 | 1 | 5 | Both ✗ |
| 4(3).jpeg | 4 | 4 | 2 | KNN ✓ |
| 5.png | 5 | 1 | 2 | Both ✗ |
| 6(1).jpg | 6 | 3 | 5 | Both ✗ |
| 6(2).png | 6 | 5 | 5 | Both ✗ |
| 6(3).jpg | 6 | 4 | 8 | Both ✗ |
| 6.jpg | 6 | 4 | 3 | Both ✗ |
| 7(2).jpg | 7 | 7 | 7 | Both ✓ |
| 7.jpg | 7 | 4 | 3 | Both ✗ |
| 8(2).png | 8 | 3 | 3 | Both ✗ |
| 8(3).png | 8 | 1 | 5 | Both ✗ |
| 8.png | 8 | 1 | 1 | Both ✗ |
| 9(2).png | 9 | 1 | 8 | Both ✗ |
| 9.png | 9 | 4 | 3 | Both ✗ |

**Results Summary:**
- KNN: 5/18 correct (27.8%)
- Logistic Regression: 4/18 correct (22.2%)

## Why the Internet Images Performed Poorly

The accuracy on internet images is much lower than on the MNIST test set, and there are several reasons for this.

First, the MNIST dataset has a very specific style. All digits are centered, the stroke width is consistent, and the images have white digits on a pure black background. Internet images often have different backgrounds, varying stroke thicknesses, and digits that aren't perfectly centered.

Second, the preprocessing might not match well. When I resize random internet images to 28x28 pixels, important details can get lost or distorted. The auto-inversion based on average brightness doesn't always work correctly for images with complex backgrounds or partial transparency.

Third, handwriting styles vary widely. The MNIST dataset was collected from a specific group of people, so the models learned patterns from that particular writing style. Internet images show many different ways people write digits, and some styles look nothing like what the models trained on.

Finally, image quality matters. Some internet images are low resolution, have compression artifacts, or include extra elements like grid lines or shadows that confuse the classifier.

This demonstrates an important concept in machine learning called domain shift. A model can perform excellently on data similar to its training set but struggle when the input distribution changes. For better real-world performance, we would need either more diverse training data or more advanced techniques like data augmentation and deep learning.

## Saved Models

All trained models are saved in the `saved_models/` folder:
- `knn_mnist.pkl`
- `logistic_regression_mnist.pkl`
