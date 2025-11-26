import sys
import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Image size (must match training)
IMG_SIZE = (64, 64)


def predict_image(image_path):
    """
    Predict whether an image is a Cat or Dog using saved models.
    """
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "saved_models")
    
    # Check if models exist
    if not os.path.exists(models_dir):
        print("Error: No saved models found!")
        print("Please run 'python cat_dog_classification.py' first to train and save models.")
        return
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'")
        return
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from '{image_path}'")
        return
    
    print("=" * 50)
    print("CAT VS DOG PREDICTION")
    print("=" * 50)
    print(f"\nImage: {image_path}")
    
    # Load saved models
    print("\nLoading models...")
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    knn_model = joblib.load(os.path.join(models_dir, "knn_model.pkl"))
    lr_model = joblib.load(os.path.join(models_dir, "logistic_regression_model.pkl"))
    print("Models loaded successfully!")
    
    # Preprocess the image (same as training)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_flattened = img_gray.flatten().reshape(1, -1)
    img_scaled = scaler.transform(img_flattened)
    
    # Make predictions
    print("\n" + "-" * 50)
    print("PREDICTIONS:")
    print("-" * 50)
    
    # KNN Prediction
    knn_pred = knn_model.predict(img_scaled)[0]
    knn_label = "Cat 🐱" if knn_pred == 0 else "Dog 🐶"
    if hasattr(knn_model, 'predict_proba'):
        knn_proba = knn_model.predict_proba(img_scaled)[0]
        knn_conf = max(knn_proba) * 100
        print(f"KNN:                 {knn_label} (Confidence: {knn_conf:.1f}%)")
    else:
        print(f"KNN:                 {knn_label}")
    
    # Logistic Regression Prediction
    lr_pred = lr_model.predict(img_scaled)[0]
    lr_label = "Cat 🐱" if lr_pred == 0 else "Dog 🐶"
    lr_proba = lr_model.predict_proba(img_scaled)[0]
    lr_conf = max(lr_proba) * 100
    print(f"Logistic Regression: {lr_label} (Confidence: {lr_conf:.1f}%)")
    
    print("-" * 50)
    
    # Final verdict (majority vote)
    if knn_pred == lr_pred:
        final_label = "Cat 🐱" if knn_pred == 0 else "Dog 🐶"
        print(f"\n✓ Both models agree: {final_label}")
    else:
        print(f"\n⚠ Models disagree! KNN says {knn_label}, LR says {lr_label}")
    
    # Display the image with predictions
    plt.figure(figsize=(8, 8))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"KNN: {knn_label} | Logistic Regression: {lr_label}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return knn_label, lr_label


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nNo image path provided!")
        print("Usage: python predict_image.py <image_path>")
        print("\nExample: python predict_image.py my_pet.jpg")
    else:
        image_path = sys.argv[1]
        predict_image(image_path)
