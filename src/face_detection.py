import cv2


MODEL_PATH = "models/face_detection_yunet_2026may.onnx"


def detect_faces(image):
    """
    Detect faces using OpenCV YuNet.
    """

    height, width = image.shape[:2]

    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (width, height),
        0.9,
        0.3,
        5000
    )

    detector.setInputSize((width, height))

    _, detections = detector.detect(image)

    if detections is None:
        return []

    return detections


if __name__ == "__main__":

    image_path = image_path = "dataset/real/real_00000.jpg"

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not load the image.")

    faces = detect_faces(image)

    print("Number of faces detected:", len(faces))

    for i, face in enumerate(faces):
        x, y, w, h = face[:4]

        print(
            f"Face {i + 1}: "
            f"x={int(x)}, "
            f"y={int(y)}, "
            f"width={int(w)}, "
            f"height={int(h)}"
        )