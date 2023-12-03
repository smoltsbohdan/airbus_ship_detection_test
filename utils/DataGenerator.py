import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, files, batch_size=16, image_size=(256, 256), shuffle=True, train_df=None):
        self.files = files
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()
        self.train_df = train_df

    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        batch_files = self.files[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_files)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            self.files = shuffle(self.files)

    def __data_generation(self, batch_files):
        X = np.zeros((self.batch_size, *self.image_size, 3))
        y = np.zeros((self.batch_size, *self.image_size, 1))

        for i, file in enumerate(batch_files):
            img = cv2.imread(f'C:/Users/smolt/PycharmProjects/airbus_ship_detection_test/{file}')
            mask = self.__masks_from_file(file)

            X[i,] = cv2.resize(img, self.image_size)
            y[i,] = np.expand_dims(cv2.resize(mask, self.image_size), axis=-1) / 255.0

        return X, y

    def __masks_from_file(self, file):
        masks = np.zeros((768, 768, 1), dtype=np.uint8)  # Змінено форму маски

        img_masks = self.train_df.loc[self.train_df['ImageId'] == file, 'EncodedPixels'].tolist()
        for mask in img_masks:
            if pd.notna(mask):
                masks[:, :, 0] += self.__rle_decode(mask)  # Використовуємо перший канал маски

        return masks

    def __rle_decode(self, mask_rle):
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(768 * 768, dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(768, 768, order='F')
