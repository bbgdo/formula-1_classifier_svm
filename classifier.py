import pickle
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

import time
start_time = time.time()

model, accuracy = pickle.load(open('f1_svm.sav', 'rb'))

categories = ['Mercedes', 'Red Bull']
IMG_SIZE = 256

image_path = "test_images/GettyImages-2182166845-6-scaled.webp"

animal_img = cv.imread(image_path, 1)


if animal_img is None:
    print(f"No such image: {image_path}")
else:
    resized_animal_img = cv.resize(animal_img, (IMG_SIZE, IMG_SIZE))
    image_vector = np.array(resized_animal_img).flatten()

    prediction = model.predict([image_vector])
    predicted_label = categories[prediction[0]]
    probabilities = model.predict_proba([image_vector])
    max_probability = max(probabilities[0]) * 100

    print(f"Prediction: {predicted_label} | Probability: {max_probability:.2f}%")

    plt.imshow(cv.cvtColor(resized_animal_img, cv.COLOR_BGR2RGB))
    plt.title(f"Prediction: {predicted_label} | Probability: {max_probability:.2f}%")
    plt.axis('off')
    plt.show()

end_time = time.time()
total_time = end_time - start_time
print(f'Model accuracy: {accuracy * 100:.2f}%')
print(f"Execution time: {total_time:.2f} seconds")
