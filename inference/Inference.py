import numpy as np
import matplotlib.pyplot as plt
import cv2
from train_model.TrainModel import train_model

model, train_df = train_model()


# Інференс для тестових зображень
random_row = train_df.sample(random_state=42)
image_id = random_row['ImageId'].values[0]
image_path = f'{image_id}'
test_image = cv2.imread(image_path)
test_image_resized = cv2.resize(test_image, (256, 256)) / 255.0
test_image_resized = np.expand_dims(test_image_resized, axis=0)

if test_image_resized.dtype != np.uint8:
    test_image_resized = (test_image_resized * 255).astype(np.uint8)

# Отримання сегментаційної маски
predictions = model.predict(test_image_resized)

if predictions.dtype != np.uint8:
    predictions = (predictions * 255).astype(np.uint8)

# Відображення оригінального зображення та сегментаційної маски поруч
plt.subplot(1, 2, 1)
plt.imshow(test_image_resized[0])
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(predictions[0, :, :, 0], cmap='gray')
plt.title('Segmentation Mask')
plt.show()

