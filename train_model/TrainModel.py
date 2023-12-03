import pandas as pd
import os
import matplotlib.pyplot as plt

from unet_model.UnetModel import unet_model
from utils.DataGenerator import DataGenerator
from sklearn.model_selection import train_test_split


def train_model():
    # Отримання абсолютного шляху до файлу
    file_path = os.path.join(os.path.dirname(__file__), "..", "data_files", "train_ship_segmentations_v2.csv")

    # Перевірка наявності файлу
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Завантаження даних
    train_df = pd.read_csv(file_path)
    train_df = train_df.sample(n=1000, random_state=42)

    # Візуалізація розподілу класів
    ship_count = train_df['EncodedPixels'].notna().sum()
    no_ship_count = len(train_df) - ship_count

    plt.bar(['With Ships', 'Without Ships'], [ship_count, no_ship_count])
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Distribution of Classes')
    plt.show()

    # Ім'я зображення та розмітка
    train_image_dir = 'train_images/'
    train_df['ImageId'] = train_df['ImageId'].apply(lambda x: os.path.join(train_image_dir, x))

    # Фільтрація зображень, на яких є кораблі
    ships = train_df['EncodedPixels'].notna()
    train_df = train_df[ships]

    # Розділення на тренувальний та валідаційний набори
    train_files, val_files = train_test_split(train_df['ImageId'], test_size=0.1, random_state=42)

    print(f"Training samples: {len(train_files)}, Validation samples: {len(val_files)}")

    # Розмірність вхідних зображень
    input_shape = (256, 256, 3)

    # Створення та компіляція моделі
    model = unet_model(input_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Створення генераторів даних
    train_generator = DataGenerator(train_files, batch_size=16, train_df=train_df)
    val_generator = DataGenerator(val_files, batch_size=16, train_df=train_df)

    # Навчання моделі
    model.fit(train_generator, epochs=2, validation_data=val_generator)
    return model, train_df
