import cv2
from tqdm.auto import tqdm
import tensorflow as tf
import random
import os
import numpy as np
def get_images_path(dir_path: str) -> list:
    # dir path would be the path of the directory that contains images

    # create img_path_list with all the file in dir_path
    img_path_list = [os.path.join(dir_path, img_name) for img_name in os.listdir(dir_path)]
    # If the path is not a file(could be a directory), ignore it
    img_path_list = [img_path for img_path in img_path_list if os.path.isfile(img_path)]

    return img_path_list
    
def process_image(img_path_list: list, img_size: tuple) -> list:
    # img_size is the width and length of the image in tuple
    processed_images = []
    
    for img_path in img_path_list:

        try:
            # read, resize image, and convert image from bgr to rgb due to the reason
            # that opencv read image in the pattern of bgr
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img - 127.5) / 127.5
            processed_images.append(img)

        except Exception as e:
            print(e)
            print(f"Can't load image from: {img_path}")

    return processed_images

class ImageLoader(tf.keras.utils.Sequence):
    def __init__(self, dir_path, img_size, batch_size):
        super().__init__()
        self.dir_path = dir_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.img_path_list = get_images_path(dir_path)
    
    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, len(self.img_path_list))
        selected_path = self.img_path_list[start_index: end_index]
        
        return np.array(process_image(selected_path, self.img_size))
    def __len__(self):
        length = len(self.img_path_list) / self.batch_size
        if length % 1:
            return int(length + 1)
        return int(length)
        
