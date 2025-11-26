import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import time

# Create directory for saved models
os.makedirs('saved_models', exist_ok=True)

def load_data():
    """Load and prepare MNIST data from CSV files"""
    print("Loading MNIST dataset...")
    
    # Load training and test data
    train_data = pd.read_csv('mnist_train.csv', header=None)
    test_data = pd.read_csv('mnist_test.csv', header=None)
    
    # First column is the label, rest are pixel values
    X_train = train_data.iloc[:, 1:].values
    y_train = train_data.iloc[:, 0].values
    X_test = test_data.iloc[:, 1:].values
    y_test = test_data.iloc[:, 0].values
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features per image: {X_train.shape[1]} (28x28 pixels)")
    print(f"Classes: {np.unique(y_train)}")
    
    return X_train, X_test, y_train, y_test

def normalize_data(X_train, X_test):
    """Normalize pixel values to 0-1 range"""
    # Simple normalization - divide by 255
    X_train_norm = X_train / 255.0
    X_test_norm = X_test / 255.0
    return X_train_norm, X_test_norm

def train_knn(X_train, X_test, y_train, y_test):
    """Train and evaluate K-Nearest Neighbors classifier"""
    print("\n" + "="*60)
    print("K-NEAREST NEIGHBORS (KNN)")
    print("="*60)
    
    # Test different K values
    k_values = [1, 3, 5, 7]
    results = []
    
    for k in k_values:
        print(f"\nTraining KNN with K={k}...")
        start_time = time.time()
        
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Predict
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            'K': k,
            'Accuracy': accuracy,
            'Time': train_time
        })
        print(f"K={k}: Accuracy = {accuracy*100:.2f}%, Time = {train_time:.2f}s")
    
    # Find best K
    best_result = max(results, key=lambda x: x['Accuracy'])
    best_k = best_result['K']
    
    print(f"\nBest K = {best_k} with accuracy = {best_result['Accuracy']*100:.2f}%")
    
    # Train final model with best K
    print(f"\nTraining final KNN model with K={best_k}...")
    knn_best = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    knn_best.fit(X_train, y_train)
    
    # Save model
    joblib.dump(knn_best, 'saved_models/knn_mnist.pkl')
    print("Model saved to saved_models/knn_mnist.pkl")
    
    return knn_best, best_result['Accuracy'], results

def train_logistic_regression(X_train, X_test, y_train, y_test):
    """Train and evaluate Logistic Regression classifier"""
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION")
    print("="*60)
    
    # Test different C values (regularization strength)
    c_values = [0.01, 0.1, 1.0]
    results = []
    
    for c in c_values:
        print(f"\nTraining Logistic Regression with C={c}...")
        start_time = time.time()
        
        lr = LogisticRegression(C=c, max_iter=1000, solver='lbfgs', 
                                multi_class='multinomial', n_jobs=-1)
        lr.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Predict
        y_pred = lr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            'C': c,
            'Accuracy': accuracy,
            'Time': train_time
        })
        print(f"C={c}: Accuracy = {accuracy*100:.2f}%, Time = {train_time:.2f}s")
    
    # Find best C
    best_result = max(results, key=lambda x: x['Accuracy'])
    best_c = best_result['C']
    
    print(f"\nBest C = {best_c} with accuracy = {best_result['Accuracy']*100:.2f}%")
    
    # Train final model with best C
    print(f"\nTraining final Logistic Regression model with C={best_c}...")
    lr_best = LogisticRegression(C=best_c, max_iter=1000, solver='lbfgs',
                                  multi_class='multinomial', n_jobs=-1)
    lr_best.fit(X_train, y_train)
    
    # Save model
    joblib.dump(lr_best, 'saved_models/logistic_regression_mnist.pkl')
    print("Model saved to saved_models/logistic_regression_mnist.pkl")
    
    return lr_best, best_result['Accuracy'], results

def train_svm(X_train, X_test, y_train, y_test):
    """Train and evaluate Support Vector Machine classifier"""
    print("\n" + "="*60)
    print("SUPPORT VECTOR MACHINE (SVM)")
    print("="*60)
    
    # Use a subset for faster training (SVM is slow on large datasets)
    # Use 10000 samples for training
    n_samples = min(10000, len(X_train))
    indices = np.random.choice(len(X_train), n_samples, replace=False)
    X_train_subset = X_train[indices]
    y_train_subset = y_train[indices]
    
    print(f"Using {n_samples} samples for SVM training (full dataset is slow)")
    
    # Test different kernels
    kernels = ['rbf', 'poly']
    results = []
    
    for kernel in kernels:
        print(f"\nTraining SVM with {kernel} kernel...")
        start_time = time.time()
        
        svm = SVC(kernel=kernel, C=1.0, gamma='scale')
        svm.fit(X_train_subset, y_train_subset)
        
        train_time = time.time() - start_time
        
        # Predict
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            'Kernel': kernel,
            'Accuracy': accuracy,
            'Time': train_time
        })
        print(f"{kernel}: Accuracy = {accuracy*100:.2f}%, Time = {train_time:.2f}s")
    
    # Find best kernel
    best_result = max(results, key=lambda x: x['Accuracy'])
    best_kernel = best_result['Kernel']
    
    print(f"\nBest kernel = {best_kernel} with accuracy = {best_result['Accuracy']*100:.2f}%")
    
    # Train final model with best kernel
    print(f"\nTraining final SVM model with {best_kernel} kernel...")
    svm_best = SVC(kernel=best_kernel, C=1.0, gamma='scale')
    svm_best.fit(X_train_subset, y_train_subset)
    
    # Save model
    joblib.dump(svm_best, 'saved_models/svm_mnist.pkl')
    print("Model saved to saved_models/svm_mnist.pkl")
    
    return svm_best, best_result['Accuracy'], results

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train and evaluate Random Forest classifier"""
    print("\n" + "="*60)
    print("RANDOM FOREST")
    print("="*60)
    
    # Test different number of trees
    n_estimators_list = [50, 100, 200]
    results = []
    
    for n_est in n_estimators_list:
        print(f"\nTraining Random Forest with {n_est} trees...")
        start_time = time.time()
        
        rf = RandomForestClassifier(n_estimators=n_est, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Predict
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results.append({
            'n_estimators': n_est,
            'Accuracy': accuracy,
            'Time': train_time
        })
        print(f"{n_est} trees: Accuracy = {accuracy*100:.2f}%, Time = {train_time:.2f}s")
    
    # Find best n_estimators
    best_result = max(results, key=lambda x: x['Accuracy'])
    best_n = best_result['n_estimators']
    
    print(f"\nBest n_estimators = {best_n} with accuracy = {best_result['Accuracy']*100:.2f}%")
    
    # Train final model
    print(f"\nTraining final Random Forest model with {best_n} trees...")
    rf_best = RandomForestClassifier(n_estimators=best_n, n_jobs=-1, random_state=42)
    rf_best.fit(X_train, y_train)
    
    # Save model
    joblib.dump(rf_best, 'saved_models/random_forest_mnist.pkl')
    print("Model saved to saved_models/random_forest_mnist.pkl")
    
    return rf_best, best_result['Accuracy'], results

def print_detailed_results(model, X_test, y_test, model_name):
    """Print detailed classification report and confusion matrix"""
    print(f"\n{'='*60}")
    print(f"DETAILED RESULTS FOR {model_name}")
    print("="*60)
    
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

def main():
    print("="*60)
    print("MNIST DIGIT CLASSIFICATION")
    print("CVI620 - Assignment 2, Question 2")
    print("="*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Normalize data
    print("\nNormalizing data...")
    X_train_norm, X_test_norm = normalize_data(X_train, X_test)
    
    # Store all results
    all_results = {}
    
    # Train all models
    knn_model, knn_acc, knn_results = train_knn(X_train_norm, X_test_norm, y_train, y_test)
    all_results['KNN'] = knn_acc
    
    lr_model, lr_acc, lr_results = train_logistic_regression(X_train_norm, X_test_norm, y_train, y_test)
    all_results['Logistic Regression'] = lr_acc
    
    svm_model, svm_acc, svm_results = train_svm(X_train_norm, X_test_norm, y_train, y_test)
    all_results['SVM'] = svm_acc
    
    rf_model, rf_acc, rf_results = train_random_forest(X_train_norm, X_test_norm, y_train, y_test)
    all_results['Random Forest'] = rf_acc
    
    # Print summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"\n{'Method':<25} {'Accuracy':<15} {'Target Met':<15}")
    print("-"*55)
    
    for method, accuracy in all_results.items():
        target_met = "YES" if accuracy >= 0.90 else "NO"
        print(f"{method:<25} {accuracy*100:.2f}%{'':<8} {target_met:<15}")
    
    # Find best model
    best_method = max(all_results, key=all_results.get)
    best_accuracy = all_results[best_method]
    
    print("-"*55)
    print(f"\nBest Method: {best_method} with {best_accuracy*100:.2f}% accuracy")
    
    if best_accuracy >= 0.90:
        print("\n*** TARGET OF 90% ACCURACY ACHIEVED! ***")
    else:
        print("\n*** Target of 90% not achieved. Consider trying other methods. ***")
    
    # Print detailed results for best model
    if best_method == 'KNN':
        print_detailed_results(knn_model, X_test_norm, y_test, 'KNN')
    elif best_method == 'Logistic Regression':
        print_detailed_results(lr_model, X_test_norm, y_test, 'Logistic Regression')
    elif best_method == 'SVM':
        print_detailed_results(svm_model, X_test_norm, y_test, 'SVM')
    else:
        print_detailed_results(rf_model, X_test_norm, y_test, 'Random Forest')
    
    print("\n" + "="*60)
    print("All models saved to 'saved_models/' directory")
    print("="*60)

if __name__ == "__main__":
    main()
