import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from train_model.TrainModel import train_model

# Інференс для тестових зображень

model, train_df = train_model()

# Визначення кількості зображень для виведення
num_images_to_display = 5

# Ініціалізація пустого списку для збереження результатів
display_images = []
display_predictions = []

# Визначення кількості зображень для виведення
num_images_to_display = 5

# Отримання випадкових індексів
random_indices = random.sample(range(len(train_df)), num_images_to_display)

# Ініціалізація списку для збереження результатів
display_images = []
display_predictions = []

# Отримання випадкових зображень та їх передбачень
for index in random_indices:
    image_id = train_df.iloc[index]['ImageId']
    image_path = f'C:/Users/smolt/PycharmProjects/airbus_ship_detection_test/{image_id}'
    test_image = cv2.imread(image_path)
    test_image_resized = cv2.resize(test_image, (256, 256)) / 255.0
    test_image_resized = np.expand_dims(test_image_resized, axis=0)

    # Отримання сегментаційної маски
    predictions = model.predict(test_image_resized)

    # Збереження зображення та його передбачення
    display_images.append(test_image_resized[0])
    display_predictions.append(predictions[0, :, :, 0])

# Відображення зображень та їх передбачень
plt.figure(figsize=(15, 5))
for i in range(num_images_to_display):
    plt.subplot(2, num_images_to_display, i + 1)
    plt.imshow(display_images[i])
    plt.title(f'Image {i + 1}')

    plt.subplot(2, num_images_to_display, i + 1 + num_images_to_display)
    plt.imshow(display_predictions[i])
    plt.title(f'Prediction {i + 1}')

plt.show()