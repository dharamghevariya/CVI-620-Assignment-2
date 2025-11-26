# Question 1: Cat vs Dog Image Classification

## Question

> For the Cat and Dog dataset provided in the Q1 folder, perform classification using all the methods you know and try to achieve the best possible result. Compare the algorithms carefully and tune the parameters so that the best result can be obtained. Save the trained model and test it on several images from the internet. Was the model able to correctly predict the images?

---

## Approach

For this classification task, I implemented two machine learning algorithms: **K-Nearest Neighbors (KNN)** and **Logistic Regression**. KNN classifies images by finding the K closest training samples and taking a majority vote, while Logistic Regression finds a linear decision boundary that best separates cats from dogs.

The dataset consists of 2000 training images (1000 cats and 1000 dogs) and 10 test images. Additionally, I downloaded 9 images from the internet (5 cats and 4 dogs) to test how well the trained models generalize to real-world images.

---

## Parameter Tuning
I tested different values of K to find the optimal number of neighbors. The validation accuracy for each K value is shown below:

| K Value | Validation Accuracy |
|---------|---------------------|
| 1       | 54.75%              |
| 3       | 56.75%              |
| 5       | 56.50%              |
| 7       | 57.00%              |
| 9       | 57.75%              |
| 11      | 57.75%              |
| 13      | 58.50%              |
| 15      | 57.50%              |
| **17**  | **58.75%** (Best)   |
| 19      | 57.50%              |
| 21      | 56.50%              |

K = 17 achieved the highest validation accuracy of 58.75%, so this was selected as the final parameter. I used odd values of K to avoid ties when voting in binary classification.

### Logistic Regression - Finding the Best C Value

For Logistic Regression, I tuned the regularization parameter C. A smaller C value means stronger regularization, which helps prevent overfitting.

| C Value   | Validation Accuracy |
|-----------|---------------------|
| **0.001** | **52.75%** (Best)   |
| 0.01      | 50.25%              |
| 0.1       | 47.25%              |
| 1         | 46.50%              |
| 10        | 48.75%              |
| 100       | 47.75%              |

Interestingly, the smallest C value (0.001) performed best, indicating that strong regularization works better for this dataset. This makes sense because with 4096 features (64x64 flattened pixels), the model is prone to overfitting without proper regularization.

---

## Model Comparison

| Algorithm           | Best Parameter | Validation Accuracy | Test Accuracy |
|---------------------|----------------|---------------------|---------------|
| **KNN**             | K = 17         | 58.75%              | **70%**       |
| Logistic Regression | C = 0.001      | 52.75%              | 60%           |

On the provided test set, KNN outperformed Logistic Regression with 70% accuracy compared to 60%.

---

## Testing on Internet Images

After training, I saved both models and tested them on 9 images downloaded from the internet. The results are shown below:

| Image       | Actual | KNN Prediction | KNN Result | LR Prediction | LR Result |
|-------------|--------|----------------|------------|---------------|-----------|
| cat1.jpg    | Cat    | Dog            | FAIL       | Cat           | PASS      |
| cat2.jpg    | Cat    | Cat            | PASS       | Cat           | PASS      |
| cat3.jpg    | Cat    | Dog            | FAIL       | Cat           | PASS      |
| cat4.jpg    | Cat    | Cat            | PASS       | Dog           | FAIL      |
| cat5.jpg    | Cat    | Cat            | PASS       | Cat           | PASS      |
| dog1.jpg    | Dog    | Dog            | PASS       | Dog           | PASS      |
| dog2.webp   | Dog    | Cat            | FAIL       | Cat           | FAIL      |
| dog3.webp   | Dog    | Dog            | PASS       | Dog           | PASS      |
| dog4.jpg    | Dog    | Cat            | FAIL       | Dog           | PASS      |

**Summary:** KNN correctly classified 5 out of 9 images (55.6%), while Logistic Regression correctly classified 7 out of 9 images (77.8%).

---

## Analysis

The results reveal an interesting finding: even though KNN performed better on the test set (70% vs 60%), Logistic Regression generalized better to internet images (77.8% vs 55.6%). This suggests that KNN may have overfit to the specific characteristics of the training and test datasets.

KNN struggled particularly with cat1.jpg and cat3.jpg, misclassifying them as dogs. This could be because these cats had features (perhaps certain fur patterns or face shapes) that were more similar to dogs in the training set. Both models failed on dog2.webp, which likely has unusual lighting, angle, or background that differs significantly from the training data.

The performance difference between the test set and internet images can be attributed to several factors. Internet images vary widely in resolution, lighting, and image quality. The backgrounds in internet images are often more complex than the training images. Additionally, some breeds or poses may not be well-represented in the training data.

---

## Conclusion

The models were partially successful in predicting internet images. Logistic Regression achieved 77.8% accuracy (7/9 correct), while KNN achieved 55.6% (5/9 correct). While this is better than random guessing (50%), there is significant room for improvement.

To achieve better results, we could use larger and more diverse training datasets, keep color information instead of converting to grayscale, apply data augmentation techniques like rotation and flipping, or use more advanced methods like Convolutional Neural Networks (CNNs) with transfer learning.

---

For setup and usage instructions, please refer to `instructions.md`.

---

**Dharam Mehulbhai Ghevariya**  
Email: dmghevariya@myseneca.ca  
Student ID: 136270220  

CVI620 - Computer Vision | Seneca Polytechnic | Fall 2025
