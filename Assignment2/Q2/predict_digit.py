import numpy as np
import cv2
import joblib
import os
import sys

def load_models():
    """Load all saved models"""
    models = {}
    model_files = {
        'KNN': 'saved_models/knn_mnist.pkl',
        'Logistic Regression': 'saved_models/logistic_regression_mnist.pkl'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"Loaded {name} model")
        else:
            print(f"Warning: {name} model not found at {path}")
    
    return models

def preprocess_image(image_path):
    """
    Preprocess an image to match MNIST format:
    - Convert to grayscale
    - Resize to 28x28
    - Invert if needed (MNIST has white digits on black background)
    - Normalize to 0-1 range
    - Flatten to 784 values
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Check if we need to invert (MNIST has white digits on black background)
    # If the image has a light background, invert it
    mean_val = np.mean(resized)
    if mean_val > 127:
        resized = 255 - resized
    
    # Normalize to 0-1 range
    normalized = resized / 255.0
    
    # Flatten to 1D array
    flattened = normalized.flatten()
    
    return flattened, resized

def predict_digit(image_path, models):
    """Predict digit using all available models"""
    print(f"\nProcessing: {image_path}")
    print("-" * 50)
    
    # Preprocess image
    features, processed_img = preprocess_image(image_path)
    features = features.reshape(1, -1)  # Reshape for sklearn
    
    # Predict with each model
    predictions = {}
    for name, model in models.items():
        pred = model.predict(features)[0]
        predictions[name] = pred
        print(f"{name}: Predicted digit = {pred}")
    
    # Majority vote
    if predictions:
        votes = list(predictions.values())
        majority = max(set(votes), key=votes.count)
        print(f"\nMajority Vote: {majority}")
    
    return predictions, processed_img

def show_image(img, title="Processed Image"):
    """Display the processed image"""
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 4))
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()
    except ImportError:
        print("Install matplotlib to view images: pip install matplotlib")

def main():
    print("=" * 50)
    print("MNIST DIGIT PREDICTION")
    print("=" * 50)
    
    # Load models
    models = load_models()
    
    if not models:
        print("\nNo models found! Run mnist_classification.py first to train models.")
        return
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        # Predict on provided image(s)
        for image_path in sys.argv[1:]:
            if os.path.exists(image_path):
                predictions, processed = predict_digit(image_path, models)
                show_image(processed, f"Processed: {os.path.basename(image_path)}")
            else:
                print(f"Image not found: {image_path}")
    else:
        # Check for images folder
        images_folder = 'images'
        if os.path.exists(images_folder):
            image_files = [f for f in os.listdir(images_folder) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if image_files:
                print(f"\nFound {len(image_files)} images in '{images_folder}/' folder")
                for img_file in sorted(image_files):
                    img_path = os.path.join(images_folder, img_file)
                    predict_digit(img_path, models)
            else:
                print(f"\nNo images found in '{images_folder}/' folder")
                print_usage()
        else:
            print_usage()

def print_usage():
    print("\nUsage:")
    print("  python predict_digit.py <image_path>")
    print("  python predict_digit.py image1.png image2.jpg ...")
    print("\nOr create an 'images/' folder with digit images to test.")
    print("\nTips for best results:")
    print("  - Use clear images of single digits")
    print("  - White or light digits on dark background work best")
    print("  - Center the digit in the image")
    print("  - Avoid too much padding or clutter")

if __name__ == "__main__":
    main()
