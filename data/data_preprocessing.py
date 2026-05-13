
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

IMG_SIZE = 256

def normalize_image(image):
    image = image.astype(np.float32) / 255.0
    return image

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = normalize_image(image)
    image = np.expand_dims(image, axis=-1)
    return image

def preprocess_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = mask.astype(np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask

class DataGenerator(Sequence):

    def __init__(self,
                 image_paths,
                 mask_paths,
                 batch_size=16,
                 shuffle=True):

        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):

        batch_indexes = self.indexes[
            index * self.batch_size:(index + 1) * self.batch_size
        ]

        batch_images = []
        batch_masks = []

        for idx in batch_indexes:

            image = preprocess_image(self.image_paths[idx])
            mask = preprocess_mask(self.mask_paths[idx])

            batch_images.append(image)
            batch_masks.append(mask)

        return np.array(batch_images), np.array(batch_masks)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
