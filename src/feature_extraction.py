import cv2
import numpy as np

from preprocessing import preprocess_image
from face_detection import detect_faces


def extract_features(image):
    """
    Extract global image + face-region features.
    """

    # -------------------------------------------------
    # Global image features
    # -------------------------------------------------

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    mean_brightness = np.mean(gray)
    brightness_std = np.std(gray)

    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = np.mean(edges > 0)

    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    blue_mean = np.mean(image[:, :, 0])
    green_mean = np.mean(image[:, :, 1])
    red_mean = np.mean(image[:, :, 2])

    blue_std = np.std(image[:, :, 0])
    green_std = np.std(image[:, :, 1])
    red_std = np.std(image[:, :, 2])

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(
        sobel_x ** 2 + sobel_y ** 2
    )

    gradient_mean = np.mean(gradient_magnitude)
    gradient_std = np.std(gradient_magnitude)

    # -------------------------------------------------
    # Face detection
    # -------------------------------------------------

    faces = detect_faces(image)

    # Default face features
    face_found = 0
    face_area_ratio = 0
    face_mean_brightness = 0
    face_brightness_std = 0
    face_sharpness = 0

    # -------------------------------------------------
    # Extract features from the largest detected face
    # -------------------------------------------------

    if len(faces) > 0:

        # Select largest face
        largest_face = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        x, y, w, h = largest_face[:4]

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        # Keep coordinates inside image boundaries
        x = max(0, x)
        y = max(0, y)

        x2 = min(image.shape[1], x + w)
        y2 = min(image.shape[0], y + h)

        face = image[y:y2, x:x2]

        if face.size > 0:

            face_gray = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2GRAY
            )

            face_found = 1

            face_area_ratio = (
                face.shape[0] * face.shape[1]
            ) / (
                image.shape[0] * image.shape[1]
            )

            face_mean_brightness = np.mean(face_gray)

            face_brightness_std = np.std(face_gray)

            face_sharpness = cv2.Laplacian(
                face_gray,
                cv2.CV_64F
            ).var()

    # -------------------------------------------------
    # Combine features
    # -------------------------------------------------

    features = [
        mean_brightness,
        brightness_std,
        edge_ratio,
        sharpness,

        blue_mean,
        green_mean,
        red_mean,

        blue_std,
        green_std,
        red_std,

        gradient_mean,
        gradient_std,

        face_found,
        face_area_ratio,
        face_mean_brightness,
        face_brightness_std,
        face_sharpness
    ]

    return np.array(features)


if __name__ == "__main__":

    image_path = "dataset/real/real_00000.jpg"

    image = preprocess_image(image_path)

    features = extract_features(image)

    print("Image processed successfully!")
    print("Number of features:", len(features))

    print("\nExtracted features:")
    print(features)