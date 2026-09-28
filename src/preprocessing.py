import cv2


def load_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not load the image.")

    return image


def resize_image(image, width=224, height=224):
    return cv2.resize(image, (width, height))


def preprocess_image(image_path):
    image = load_image(image_path)
    image = resize_image(image)

    return image


if __name__ == "__main__":
    image_path = "dataset/real/Screenshot 2026-02-07 221503.png"

    image = preprocess_image(image_path)

    print("Image loaded successfully!")
    print("Image shape:", image.shape)



    