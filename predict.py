import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Parameters
img_size = 224

# Load the model
model = load_model('dental_xray_classifier.h5')

def predict_image(model, img_path, img_size):
    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    resized_array = cv2.resize(img_array, (img_size, img_size))
    normalized_array = resized_array / 255.0
    reshaped_array = normalized_array.reshape(-1, img_size, img_size, 1)
    prediction = model.predict(reshaped_array)
    return 'child' if prediction[0][0] < 0.5 else 'adult'

# Example prediction
# img_path = 'untested_png/child/child (1).png'  # Replace with the actual image path
# prediction = predict_image(model, img_path, img_size)
# print(f'The x-ray is of a {prediction}.')

# write a for loop from 1 to 10
for i in range(1, 11):
    img_path = f'./untested_png/adult/adult ({i}).png'  # Replace with the actual image path
    prediction = predict_image(model, img_path, img_size)
    print(f'The x-ray number {i} is of a {prediction}.')
for i in range(1, 11):
    img_path = f'./untested_png/child/child ({i}).png'  # Replace with the actual image path
    prediction = predict_image(model, img_path, img_size)
    print(f'The x-ray number {i} is of a {prediction}.')
