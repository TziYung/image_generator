import cv2
from tqdm.auto import tqdm
import random
def image_process(dir_path: str, img_size: tuple) -> list:
    # dir path would be the path of the directory that contains images
    # img_size is the width and length of the image in tuple

    print(f"Loading from {dir_path}")
    processed_images = []
    
    for img_name in tqdm(os.listdir(dir_path)):
        img_path = os.path.join(dir_path, img_name)

        # If the path is not a file(could be a directory), ignore it
        if os.path.isfile(img_path) == False:
            continue

        try:
            # read, resize image, and convert image from bgr to rgb due to the reason
            # that opencv read image in the pattern of bgr
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_images.append(img)

        except Exception as e:
            print(e)
            print(f"Can't load image from: {path}")
    return processed_images


