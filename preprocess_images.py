import os
from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)
from tensorflow.keras.applications import vgg16

def get_preprocessed_images(
        image_paths: list,
        image_size: int = (224,224)
):
    print("Preparing images...")

    images = []
    labels = []

    for imagePath in image_paths:
        path_parts = imagePath.split(os.path.sep)

        label = path_parts[1]

        image = load_img(imagePath, target_size=image_size)

        images.append(image)
        labels.append(label)

    preprocess_images = []

    for image in images:
        image = img_to_array(image)
        image = vgg16.preprocess_input(image)
        image = img_to_array(image)

        preprocess_images.append(image)

    print("Images are ready.")
    print(f"Total number of images: {len(images)} \n")

    return preprocess_images, labels