import numpy as np
import os
import cv2
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Image size (must be same for training and prediction)
IMG_SIZE = (64, 64)

# ============================================================================
# STEP 1: LOAD AND PREPROCESS IMAGES
# ============================================================================

def load_images(data_dir, img_size=(64, 64)):
    """
    Load images from directory and convert to feature vectors.
    
    Parameters:
    - data_dir: Path to directory containing Cat and Dog folders
    - img_size: Tuple (width, height) to resize images
    
    Returns:
    - X: Feature matrix (flattened pixel values)
    - y: Labels (0 for Cat, 1 for Dog)
    """
    X = []
    y = []
    
    # Load Cat images (label = 0)
    cat_dir = os.path.join(data_dir, 'Cat')
    print(f"Loading Cat images from: {cat_dir}")
    cat_count = 0
    for filename in os.listdir(cat_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(cat_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # Resize and convert to grayscale for simplicity
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Flatten the image to 1D array
                X.append(img.flatten())
                y.append(0)  # Cat = 0
                cat_count += 1
    print(f"Loaded {cat_count} Cat images")
    
    # Load Dog images (label = 1)
    dog_dir = os.path.join(data_dir, 'Dog')
    print(f"Loading Dog images from: {dog_dir}")
    dog_count = 0
    for filename in os.listdir(dog_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(dog_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                X.append(img.flatten())
                y.append(1)  # Dog = 1
                dog_count += 1
    print(f"Loaded {dog_count} Dog images")
    
    return np.array(X), np.array(y)


# ============================================================================
# FUNCTION TO PREDICT ON A NEW IMAGE
# ============================================================================

def predict_single_image(image_path, model, scaler, model_name="Model"):
    """
    Predict whether an image is a Cat or Dog using a trained model.
    
    Parameters:
    - image_path: Path to the image file
    - model: Trained model (KNN or Logistic Regression)
    - scaler: Fitted StandardScaler
    - model_name: Name of the model for display
    
    Returns:
    - prediction: 'Cat' or 'Dog'
    """
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Resize and convert to grayscale (same as training)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Flatten to 1D array
    img_flattened = img_gray.flatten().reshape(1, -1)
    
    # Scale the features
    img_scaled = scaler.transform(img_flattened)
    
    # Make prediction
    prediction = model.predict(img_scaled)[0]
    label = "Cat" if prediction == 0 else "Dog"
    
    # Get prediction probability if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(img_scaled)[0]
        confidence = max(proba) * 100
        print(f"\n{model_name} Prediction: {label} (Confidence: {confidence:.1f}%)")
    else:
        print(f"\n{model_name} Prediction: {label}")
    
    # Display the image with prediction
    plt.figure(figsize=(6, 6))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"{model_name} Prediction: {label}", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()
    
    return label


def load_models_and_predict(image_path):
    """
    Load saved models and predict on a new image.
    
    Parameters:
    - image_path: Path to the image to classify
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "saved_models")
    
    # Check if models exist
    if not os.path.exists(models_dir):
        print("Error: No saved models found. Please run training first.")
        return
    
    # Load the scaler and models
    print("Loading saved models...")
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    knn_model = joblib.load(os.path.join(models_dir, "knn_model.pkl"))
    lr_model = joblib.load(os.path.join(models_dir, "logistic_regression_model.pkl"))
    print("Models loaded successfully!")
    
    # Make predictions with both models
    print("\n" + "=" * 50)
    print("PREDICTIONS ON NEW IMAGE")
    print("=" * 50)
    
    knn_pred = predict_single_image(image_path, knn_model, scaler, "KNN")
    lr_pred = predict_single_image(image_path, lr_model, scaler, "Logistic Regression")
    
    return knn_pred, lr_pred


# ============================================================================
# STEP 2: MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Set paths (relative to the script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, "train")
    test_dir = os.path.join(script_dir, "test")
    models_dir = os.path.join(script_dir, "saved_models")
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Image size for resizing (smaller = faster, larger = more detail)
    # Note: IMG_SIZE is defined globally at the top of the file
    
    print("=" * 60)
    print("CAT VS DOG CLASSIFICATION")
    print("=" * 60)
    
    # Load training data
    print("\n[1] Loading Training Data...")
    X_train_full, y_train_full = load_images(train_dir, IMG_SIZE)
    print(f"Training data shape: {X_train_full.shape}")
    print(f"Number of features per image: {X_train_full.shape[1]}")
    
    # Load test data
    print("\n[2] Loading Test Data...")
    X_test, y_test = load_images(test_dir, IMG_SIZE)
    print(f"Test data shape: {X_test.shape}")
    
    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # ========================================================================
    # STEP 3: FEATURE SCALING (Important for KNN and Logistic Regression)
    # ========================================================================
    print("\n[3] Scaling Features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    print("Feature scaling complete!")
    
    # ========================================================================
    # STEP 4: K-NEAREST NEIGHBORS (KNN) CLASSIFICATION
    # ========================================================================
    print("\n" + "=" * 60)
    print("K-NEAREST NEIGHBORS (KNN) CLASSIFICATION")
    print("=" * 60)
    
    # 4.1: Find the best K value
    print("\n[4.1] Finding the best K value...")
    k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    knn_accuracies = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        val_pred = knn.predict(X_val_scaled)
        acc = accuracy_score(y_val, val_pred)
        knn_accuracies.append(acc)
        print(f"  K = {k:2d} --> Validation Accuracy: {acc * 100:.2f}%")
    
    # Find best K
    best_k_idx = np.argmax(knn_accuracies)
    best_k = k_values[best_k_idx]
    best_knn_acc = knn_accuracies[best_k_idx]
    print(f"\n*** Best K = {best_k} with Validation Accuracy: {best_knn_acc * 100:.2f}% ***")
    
    # 4.2: Train final KNN model with best K
    print(f"\n[4.2] Training final KNN model with K = {best_k}...")
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_train_full_scaled, y_train_full)
    
    # Evaluate on test set
    knn_test_pred = knn_final.predict(X_test_scaled)
    knn_test_acc = accuracy_score(y_test, knn_test_pred)
    print(f"\nKNN Test Accuracy: {knn_test_acc * 100:.2f}%")
    print("\nKNN Classification Report:")
    print(classification_report(y_test, knn_test_pred, target_names=['Cat', 'Dog']))
    
    # ========================================================================
    # STEP 5: LOGISTIC REGRESSION CLASSIFICATION
    # ========================================================================
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION CLASSIFICATION")
    print("=" * 60)
    
    # 5.1: Find the best C value (regularization parameter)
    print("\n[5.1] Finding the best C value (regularization strength)...")
    c_values = [0.001, 0.01, 0.1, 1, 10, 100]
    lr_accuracies = []
    
    for c in c_values:
        lr = LogisticRegression(C=c, max_iter=1000, solver='lbfgs')
        lr.fit(X_train_scaled, y_train)
        val_pred = lr.predict(X_val_scaled)
        acc = accuracy_score(y_val, val_pred)
        lr_accuracies.append(acc)
        print(f"  C = {c:6.3f} --> Validation Accuracy: {acc * 100:.2f}%")
    
    # Find best C
    best_c_idx = np.argmax(lr_accuracies)
    best_c = c_values[best_c_idx]
    best_lr_acc = lr_accuracies[best_c_idx]
    print(f"\n*** Best C = {best_c} with Validation Accuracy: {best_lr_acc * 100:.2f}% ***")
    
    # 5.2: Train final Logistic Regression model with best C
    print(f"\n[5.2] Training final Logistic Regression model with C = {best_c}...")
    lr_final = LogisticRegression(C=best_c, max_iter=1000, solver='lbfgs')
    lr_final.fit(X_train_full_scaled, y_train_full)
    
    # Evaluate on test set
    lr_test_pred = lr_final.predict(X_test_scaled)
    lr_test_acc = accuracy_score(y_test, lr_test_pred)
    print(f"\nLogistic Regression Test Accuracy: {lr_test_acc * 100:.2f}%")
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, lr_test_pred, target_names=['Cat', 'Dog']))
    
    # ========================================================================
    # STEP 6: COMPARISON SUMMARY
    # ========================================================================
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Algorithm':<25} {'Best Parameters':<20} {'Test Accuracy':<15}")
    print("-" * 60)
    print(f"{'KNN':<25} {'K = ' + str(best_k):<20} {knn_test_acc * 100:.2f}%")
    print(f"{'Logistic Regression':<25} {'C = ' + str(best_c):<20} {lr_test_acc * 100:.2f}%")
    print("-" * 60)
    
    # Determine winner
    if knn_test_acc > lr_test_acc:
        winner = "KNN"
        winner_acc = knn_test_acc
    else:
        winner = "Logistic Regression"
        winner_acc = lr_test_acc
    
    print(f"\n*** BEST MODEL: {winner} with {winner_acc * 100:.2f}% accuracy ***")
    
    # ========================================================================
    # STEP 7: SAVE TRAINED MODELS
    # ========================================================================
    print("\n" + "=" * 60)
    print("SAVING TRAINED MODELS")
    print("=" * 60)
    
    # Save the scaler (needed for preprocessing new images)
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save KNN model
    knn_path = os.path.join(models_dir, "knn_model.pkl")
    joblib.dump(knn_final, knn_path)
    print(f"KNN model saved to: {knn_path}")
    
    # Save Logistic Regression model
    lr_path = os.path.join(models_dir, "logistic_regression_model.pkl")
    joblib.dump(lr_final, lr_path)
    print(f"Logistic Regression model saved to: {lr_path}")
    
    print("\nAll models saved successfully!")
    
    # ========================================================================
    # STEP 8: VISUALIZATION
    # ========================================================================
    print("\n[8] Creating visualizations...")
    
    # Plot 1: KNN accuracy vs K values
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(k_values, [acc * 100 for acc in knn_accuracies], 'bo-', linewidth=2, markersize=8)
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best K = {best_k}')
    plt.xlabel('K (Number of Neighbors)', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.title('KNN: Accuracy vs K Value', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Logistic Regression accuracy vs C values
    plt.subplot(1, 3, 2)
    plt.semilogx(c_values, [acc * 100 for acc in lr_accuracies], 'go-', linewidth=2, markersize=8)
    plt.axvline(x=best_c, color='r', linestyle='--', label=f'Best C = {best_c}')
    plt.xlabel('C (Regularization)', fontsize=12)
    plt.ylabel('Validation Accuracy (%)', fontsize=12)
    plt.title('Logistic Regression: Accuracy vs C Value', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Comparison bar chart
    plt.subplot(1, 3, 3)
    algorithms = ['KNN', 'Logistic\nRegression']
    accuracies = [knn_test_acc * 100, lr_test_acc * 100]
    colors = ['blue', 'green']
    bars = plt.bar(algorithms, accuracies, color=colors, edgecolor='black', linewidth=2)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Algorithm Comparison', fontsize=14)
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(train_dir, '..', 'classification_results.png'), dpi=150)
    plt.show()
    
    print("\nVisualization saved as 'classification_results.png'")
    print("\n" + "=" * 60)
    print("CLASSIFICATION COMPLETE!")
    print("=" * 60)
    
    # ========================================================================
    # STEP 9: TEST PREDICTION ON A NEW IMAGE (Optional Demo)
    # ========================================================================
    print("\n" + "=" * 60)
    print("HOW TO USE SAVED MODELS FOR PREDICTION")
    print("=" * 60)
    print("""
To predict on a new image (e.g., from the internet), you can:

1. Save your image to the Q1 folder (e.g., 'my_pet.jpg')

2. Run in Python:
   
   from cat_dog_classification import load_models_and_predict
   load_models_and_predict("my_pet.jpg")

   OR run this script with an image path:
   
   python predict_image.py path/to/your/image.jpg
""")
    
    # Demo: Predict on a test image if available
    demo_images = [
        os.path.join(test_dir, "Cat", "Cat (1).jpg"),
        os.path.join(test_dir, "Dog", "Dog (1).jpg")
    ]
    
    print("\n--- Demo Predictions on Test Images ---")
    for demo_img in demo_images:
        if os.path.exists(demo_img):
            predict_single_image(demo_img, knn_final, scaler, "KNN")
            predict_single_image(demo_img, lr_final, scaler, "Logistic Regression")
