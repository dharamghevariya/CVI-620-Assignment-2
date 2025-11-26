import os
import cv2
import joblib
import numpy as np

IMG_SIZE = (64, 64)
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, 'saved_models')
images_dir = os.path.join(script_dir, 'images')

# Load models
scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
knn_model = joblib.load(os.path.join(models_dir, 'knn_model.pkl'))
lr_model = joblib.load(os.path.join(models_dir, 'logistic_regression_model.pkl'))

print('=' * 80)
print('TESTING SAVED MODELS ON INTERNET IMAGES')
print('=' * 80)

# Test all images
images = ['cat1.jpg', 'cat2.jpg', 'cat3.jpg', 'cat4.jpg', 'cat5.jpg', 
          'dog1.jpg', 'dog2.webp', 'dog3.webp', 'dog4.jpg']

print(f"\n{'Image':<15} {'Actual':<10} {'KNN Pred':<12} {'KNN':<6} {'LR Pred':<12} {'LR':<6}")
print('-' * 80)

knn_correct = 0
lr_correct = 0
total = 0

for img_name in images:
    img_path = os.path.join(images_dir, img_name)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f'{img_name:<15} Could not load image')
        continue
    
    # Preprocess
    img_resized = cv2.resize(img, IMG_SIZE)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_flattened = img_gray.flatten().reshape(1, -1)
    img_scaled = scaler.transform(img_flattened)
    
    # Predict
    knn_pred = knn_model.predict(img_scaled)[0]
    lr_pred = lr_model.predict(img_scaled)[0]
    
    knn_label = 'Cat' if knn_pred == 0 else 'Dog'
    lr_label = 'Cat' if lr_pred == 0 else 'Dog'
    
    # Get actual label from filename
    actual = 'Cat' if 'cat' in img_name.lower() else 'Dog'
    
    knn_status = 'PASS' if knn_label == actual else 'FAIL'
    lr_status = 'PASS' if lr_label == actual else 'FAIL'
    
    if knn_label == actual:
        knn_correct += 1
    if lr_label == actual:
        lr_correct += 1
    total += 1
    
    print(f'{img_name:<15} {actual:<10} {knn_label:<12} {knn_status:<6} {lr_label:<12} {lr_status:<6}')

print('-' * 80)
print(f'\nKNN Accuracy on Internet Images:                 {knn_correct}/{total} ({knn_correct/total*100:.1f}%)')
print(f'Logistic Regression Accuracy on Internet Images: {lr_correct}/{total} ({lr_correct/total*100:.1f}%)')
print('=' * 80)
