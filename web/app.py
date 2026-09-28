import os
import sys
import traceback

import cv2
import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, jsonify


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

MODEL_DIR = os.path.join(BASE_DIR, "models")
AUDIO_MODEL_DIR = os.path.join(BASE_DIR, "audio", "models")
TEXT_MODEL_DIR = os.path.join(BASE_DIR, "text", "models")

UPLOAD_DIR = os.path.join(BASE_DIR, "web", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

IMAGE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "deepfake_model_group_split.pkl"
)

AUDIO_MODEL_PATH = os.path.join(
    AUDIO_MODEL_DIR,
    "audio_model.pkl"
)

TEXT_MODEL_PATH = os.path.join(
    TEXT_MODEL_DIR,
    "text_model.pkl"
)

FACE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "face_detection_yunet_2026may.onnx"
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "web", "templates"),
    static_folder=os.path.join(BASE_DIR, "web", "static")
)

app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


# ============================================================
# DATABASE
# ============================================================

# IMPORTANT:
# Database credentials are read from environment variables.
# Do NOT put your real MySQL password in this file.

DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "deepfake_detection")
}


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

try:
    from feature_extraction import extract_features
except Exception:
    extract_features = None

try:
    from video_detector import detect_video
except Exception:
    detect_video = None

try:
    from audio_detector import extract_audio_features
except Exception:
    extract_audio_features = None


# ============================================================
# LOAD IMAGE MODEL
# ============================================================

image_model = None

try:
    image_model = joblib.load(IMAGE_MODEL_PATH)

    print("IMAGE MODEL LOADED SUCCESSFULLY")
    print(f"Model: {IMAGE_MODEL_PATH}")

except Exception as e:
    print("IMAGE MODEL LOAD ERROR:")
    print(e)


# ============================================================
# LOAD AUDIO MODEL
# ============================================================

audio_model = None
audio_feature_names = []

try:
    audio_model_data = joblib.load(AUDIO_MODEL_PATH)

    if isinstance(audio_model_data, dict):
        audio_model = audio_model_data["model"]
        audio_feature_names = audio_model_data.get(
            "feature_names",
            []
        )
    else:
        audio_model = audio_model_data

    print("AUDIO MODEL LOADED SUCCESSFULLY")
    print(f"Model: {AUDIO_MODEL_PATH}")
    print(f"Audio features: {len(audio_feature_names)}")

except Exception as e:
    print("AUDIO MODEL LOAD ERROR:")
    print(e)


# ============================================================
# LOAD TEXT MODEL
# ============================================================

text_model = None
text_vectorizer = None

try:
    text_model_data = joblib.load(TEXT_MODEL_PATH)

    if isinstance(text_model_data, dict):
        text_model = text_model_data["model"]
        text_vectorizer = text_model_data["vectorizer"]
    else:
        text_model = text_model_data

    print("TEXT MODEL LOADED SUCCESSFULLY")
    print(f"Model: {TEXT_MODEL_PATH}")
    print("Text features: TF-IDF")

except Exception as e:
    print("TEXT MODEL LOAD ERROR:")
    print(e)


# ============================================================
# LOAD YUNET FACE DETECTOR
# ============================================================

face_detector = None

try:
    face_detector = cv2.FaceDetectorYN.create(
        FACE_MODEL_PATH,
        "",
        (320, 320),
        0.6,
        0.3,
        5000
    )

    print("YuNet face detector loaded successfully.")

except Exception as e:
    print("YuNet face detector load error:")
    print(e)


# ============================================================
# DATABASE FUNCTION
# ============================================================

def get_db_connection():

    import mysql.connector

    connection = mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"]
    )

    return connection


# ============================================================
# SAVE HISTORY
# ============================================================

def save_history(
    filename,
    media_type,
    prediction,
    confidence,
    real_probability,
    fake_probability
):

    try:

        connection = get_db_connection()
        cursor = connection.cursor()

        query = """
        INSERT INTO detection_history
        (
            filename,
            media_type,
            prediction,
            confidence,
            real_probability,
            fake_probability
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        values = (
            filename,
            media_type,
            prediction,
            confidence,
            real_probability,
            fake_probability
        )

        cursor.execute(query, values)

        connection.commit()

        cursor.close()
        connection.close()

    except Exception as e:

        print("DATABASE SAVE ERROR:")
        print(e)


# ============================================================
# IMAGE FEATURE EXTRACTION
# ============================================================

def extract_image_features(image):

    if image is None:
        raise ValueError("Image could not be loaded.")

    image = cv2.resize(
        image,
        (224, 224)
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Basic brightness
    # --------------------------------------------------------

    mean_brightness = float(
        np.mean(gray)
    )

    brightness_std = float(
        np.std(gray)
    )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_ratio = float(
        np.mean(edges > 0)
    )

    # --------------------------------------------------------
    # Sharpness
    # --------------------------------------------------------

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    # --------------------------------------------------------
    # Color statistics
    # --------------------------------------------------------

    blue_mean = float(
        np.mean(image[:, :, 0])
    )

    green_mean = float(
        np.mean(image[:, :, 1])
    )

    red_mean = float(
        np.mean(image[:, :, 2])
    )

    blue_std = float(
        np.std(image[:, :, 0])
    )

    green_std = float(
        np.std(image[:, :, 1])
    )

    red_std = float(
        np.std(image[:, :, 2])
    )

    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    grad_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    grad_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    gradient_magnitude = np.sqrt(
        grad_x ** 2 +
        grad_y ** 2
    )

    gradient_mean = float(
        np.mean(gradient_magnitude)
    )

    gradient_std = float(
        np.std(gradient_magnitude)
    )

    # --------------------------------------------------------
    # Face detection
    # --------------------------------------------------------

    face_found = 0
    face_area_ratio = 0.0
    face_mean_brightness = 0.0
    face_brightness_std = 0.0
    face_sharpness = 0.0

    if face_detector is not None:

        try:

            height, width = image.shape[:2]

            face_detector.setInputSize(
                (width, height)
            )

            _, faces = face_detector.detect(
                image
            )

            if faces is not None and len(faces) > 0:

                face_found = 1

                # First detected face
                face = faces[0]

                x = max(
                    0,
                    int(face[0])
                )

                y = max(
                    0,
                    int(face[1])
                )

                w = max(
                    1,
                    int(face[2])
                )

                h = max(
                    1,
                    int(face[3])
                )

                x2 = min(
                    width,
                    x + w
                )

                y2 = min(
                    height,
                    y + h
                )

                face_crop = image[
                    y:y2,
                    x:x2
                ]

                if face_crop.size > 0:

                    face_gray = cv2.cvtColor(
                        face_crop,
                        cv2.COLOR_BGR2GRAY
                    )

                    face_area_ratio = float(
                        (
                            face_crop.shape[0] *
                            face_crop.shape[1]
                        )
                        /
                        (
                            height *
                            width
                        )
                    )

                    face_mean_brightness = float(
                        np.mean(face_gray)
                    )

                    face_brightness_std = float(
                        np.std(face_gray)
                    )

                    face_sharpness = float(
                        cv2.Laplacian(
                            face_gray,
                            cv2.CV_64F
                        ).var()
                    )

        except Exception as e:

            print("FACE FEATURE ERROR:")
            print(e)


    # --------------------------------------------------------
    # EXACT 17 FEATURES
    # --------------------------------------------------------

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

    return features


# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_image(image_path):

    if image_model is None:
        raise RuntimeError(
            "Image model is not loaded."
        )

    image = cv2.imread(
        image_path
    )

    if image is None:
        raise ValueError(
            "Could not read image file."
        )

    print("Image loaded successfully.")

    features = extract_image_features(
        image
    )

    print(
        f"Extracted {len(features)} image features."
    )

    # --------------------------------------------------------
    # Get feature names from trained model
    # --------------------------------------------------------

    if hasattr(
        image_model,
        "feature_names_in_"
    ):

        feature_names = list(
            image_model.feature_names_in_
        )

    else:

        feature_names = [
            "mean_brightness",
            "brightness_std",
            "edge_ratio",
            "sharpness",
            "blue_mean",
            "green_mean",
            "red_mean",
            "blue_std",
            "green_std",
            "red_std",
            "gradient_mean",
            "gradient_std",
            "face_found",
            "face_area_ratio",
            "face_mean_brightness",
            "face_brightness_std",
            "face_sharpness"
        ]

    if len(feature_names) != len(features):

        raise ValueError(
            f"Feature mismatch. "
            f"Model expects {len(feature_names)} "
            f"features but extractor generated "
            f"{len(features)}."
        )

    features_df = pd.DataFrame(
        [features],
        columns=feature_names
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    raw_prediction = image_model.predict(
        features_df
    )[0]

    probabilities = image_model.predict_proba(
        features_df
    )[0]

    print(
        f"Raw image prediction: {raw_prediction}"
    )

    print(
        f"Image probabilities: {probabilities}"
    )

    # --------------------------------------------------------
    # Model may return REAL / FAKE strings
    # or 0 / 1 numeric labels.
    # --------------------------------------------------------

    if isinstance(
        raw_prediction,
        str
    ):

        prediction_text = (
            raw_prediction
            .strip()
            .upper()
        )

        if prediction_text not in [
            "REAL",
            "FAKE"
        ]:

            raise ValueError(
                f"Unknown image prediction label: "
                f"{raw_prediction}"
            )

        prediction = prediction_text

    else:

        prediction_value = int(
            raw_prediction
        )

        if prediction_value == 0:
            prediction = "REAL"

        elif prediction_value == 1:
            prediction = "FAKE"

        else:
            raise ValueError(
                f"Unknown numeric image prediction: "
                f"{prediction_value}"
            )

    # --------------------------------------------------------
    # Probability handling
    # --------------------------------------------------------

    if hasattr(
        image_model,
        "classes_"
    ):

        classes = list(
            image_model.classes_
        )

        real_probability = 0.0
        fake_probability = 0.0

        for index, class_value in enumerate(classes):

            class_name = str(
                class_value
            ).strip().upper()

            if class_name == "REAL":

                real_probability = (
                    float(probabilities[index])
                    * 100
                )

            elif class_name == "FAKE":

                fake_probability = (
                    float(probabilities[index])
                    * 100
                )

            elif class_name == "0":

                real_probability = (
                    float(probabilities[index])
                    * 100
                )

            elif class_name == "1":

                fake_probability = (
                    float(probabilities[index])
                    * 100
                )

    else:

        # Fallback
        real_probability = (
            float(probabilities[0])
            * 100
        )

        fake_probability = (
            float(probabilities[1])
            * 100
        )

    confidence = max(
        real_probability,
        fake_probability
    )

    return {
        "prediction": prediction,
        "confidence": round(
            confidence,
            2
        ),
        "real_probability": round(
            real_probability,
            2
        ),
        "fake_probability": round(
            fake_probability,
            2
        )
    }


# ============================================================
# AUDIO PREDICTION
# ============================================================

def predict_audio(audio_path):

    if audio_model is None:
        raise RuntimeError(
            "Audio model is not loaded."
        )

    if extract_audio_features is None:
        raise RuntimeError(
            "Audio feature extractor is not available."
        )

    features = extract_audio_features(
        audio_path
    )

    if len(features) != len(
        audio_feature_names
    ):

        raise ValueError(
            f"Audio feature mismatch. "
            f"Model expects {len(audio_feature_names)} "
            f"features but extractor generated "
            f"{len(features)}."
        )

    features_df = pd.DataFrame(
        [features],
        columns=audio_feature_names
    )

    raw_prediction = audio_model.predict(
        features_df
    )[0]

    probabilities = audio_model.predict_proba(
        features_df
    )[0]

    prediction_value = int(
        raw_prediction
    )

    # Audio dataset mapping:
    # 0 = REAL
    # 1 = FAKE

    if prediction_value == 0:
        prediction = "REAL"

    else:
        prediction = "FAKE"

    real_probability = (
        float(probabilities[0])
        * 100
    )

    fake_probability = (
        float(probabilities[1])
        * 100
    )

    confidence = max(
        real_probability,
        fake_probability
    )

    return {
        "prediction": prediction,
        "confidence": round(
            confidence,
            2
        ),
        "real_probability": round(
            real_probability,
            2
        ),
        "fake_probability": round(
            fake_probability,
            2
        )
    }


# ============================================================
# TEXT PREDICTION
# ============================================================

def predict_text(text):

    if text_model is None:
        raise RuntimeError(
            "Text model is not loaded."
        )

    if text_vectorizer is None:
        raise RuntimeError(
            "Text vectorizer is not loaded."
        )

    text = text.strip()

    if not text:
        raise ValueError(
            "Text cannot be empty."
        )

    vectorized_text = text_vectorizer.transform(
        [text]
    )

    raw_prediction = text_model.predict(
        vectorized_text
    )[0]

    probabilities = text_model.predict_proba(
        vectorized_text
    )[0]

    prediction_value = int(
        raw_prediction
    )

    # Text dataset mapping:
    # 0 = REAL / HUMAN
    # 1 = FAKE / AI

    if prediction_value == 0:
        prediction = "REAL"

    else:
        prediction = "FAKE"

    real_probability = (
        float(probabilities[0])
        * 100
    )

    fake_probability = (
        float(probabilities[1])
        * 100
    )

    confidence = max(
        real_probability,
        fake_probability
    )

    return {
        "prediction": prediction,
        "confidence": round(
            confidence,
            2
        ),
        "real_probability": round(
            real_probability,
            2
        ),
        "fake_probability": round(
            fake_probability,
            2
        )
    }


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# IMAGE API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "error": "No image file provided."
            }), 400

        file = request.files["file"]

        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "No image selected."
            }), 400

        filename = file.filename

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        print(
            f"Received image: {filename}"
        )

        result = predict_image(
            filepath
        )

        print(
            f"IMAGE RESULT: {result}"
        )

        save_history(
            filename=filename,
            media_type="IMAGE",
            prediction=result["prediction"],
            confidence=result["confidence"],
            real_probability=result["real_probability"],
            fake_probability=result["fake_probability"]
        )

        return jsonify({
            "success": True,
            "filename": filename,
            "media_type": "IMAGE",
            **result
        })

    except Exception as e:

        print(
            "IMAGE ANALYSIS ERROR:"
        )

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# VIDEO API
# ============================================================

@app.route(
    "/predict-video",
    methods=["POST"]
)
def predict_video():

    try:

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "error": "No video file provided."
            }), 400

        file = request.files["file"]

        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "No video selected."
            }), 400

        filename = file.filename

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        if detect_video is None:

            raise RuntimeError(
                "Video detector is not available."
            )

        result = detect_video(
            filepath
        )

        prediction = result.get(
            "prediction",
            result.get("label", "UNKNOWN")
        )

        confidence = float(
            result.get(
                "confidence",
                0
            )
        )

        real_probability = float(
            result.get(
                "real_probability",
                0
            )
        )

        fake_probability = float(
            result.get(
                "fake_probability",
                0
            )
        )

        save_history(
            filename=filename,
            media_type="VIDEO",
            prediction=prediction,
            confidence=confidence,
            real_probability=real_probability,
            fake_probability=fake_probability
        )

        return jsonify({
            "success": True,
            "filename": filename,
            "media_type": "VIDEO",
            **result
        })

    except Exception as e:

        print(
            "VIDEO ANALYSIS ERROR:"
        )

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# AUDIO API
# ============================================================

@app.route(
    "/predict-audio",
    methods=["POST"]
)
def predict_audio_route():

    try:

        if "file" not in request.files:

            return jsonify({
                "success": False,
                "error": "No audio file provided."
            }), 400

        file = request.files["file"]

        if file.filename == "":

            return jsonify({
                "success": False,
                "error": "No audio selected."
            }), 400

        filename = file.filename

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        print(
            f"Received audio: {filename}"
        )

        result = predict_audio(
            filepath
        )

        print(
            f"AUDIO RESULT: {result}"
        )

        save_history(
            filename=filename,
            media_type="AUDIO",
            prediction=result["prediction"],
            confidence=result["confidence"],
            real_probability=result["real_probability"],
            fake_probability=result["fake_probability"]
        )

        return jsonify({
            "success": True,
            "filename": filename,
            "media_type": "AUDIO",
            **result
        })

    except Exception as e:

        print(
            "AUDIO ANALYSIS ERROR:"
        )

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# TEXT API
# ============================================================

@app.route(
    "/predict-text",
    methods=["POST"]
)
def predict_text_route():

    try:

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({
                "success": False,
                "error": "No JSON data provided."
            }), 400

        text = data.get(
            "text",
            ""
        )

        if not text.strip():

            return jsonify({
                "success": False,
                "error": "Text cannot be empty."
            }), 400

        print(
            "Received text analysis request."
        )

        result = predict_text(
            text
        )

        print(
            f"TEXT RESULT: {result}"
        )

        filename = "text_input"

        save_history(
            filename=filename,
            media_type="TEXT",
            prediction=result["prediction"],
            confidence=result["confidence"],
            real_probability=result["real_probability"],
            fake_probability=result["fake_probability"]
        )

        return jsonify({
            "success": True,
            "filename": filename,
            "media_type": "TEXT",
            **result
        })

    except Exception as e:

        print(
            "TEXT ANALYSIS ERROR:"
        )

        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "success": True,
        "image_model_loaded": image_model is not None,
        "audio_model_loaded": audio_model is not None,
        "text_model_loaded": text_model is not None,
        "face_detector_loaded": face_detector is not None
    })


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    records = []

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                filename,
                media_type,
                prediction,
                confidence,
                real_probability,
                fake_probability,
                analyzed_at
            FROM detection_history
            ORDER BY analyzed_at DESC
            """
        )

        records = cursor.fetchall()

        cursor.close()
        connection.close()

    except Exception as e:

        print(
            "HISTORY ERROR:"
        )

        print(e)

    return render_template(
        "history.html",
        records=records
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    records = []

    image_count = 0
    video_count = 0
    audio_count = 0
    text_count = 0

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                filename,
                media_type,
                prediction,
                confidence,
                real_probability,
                fake_probability,
                analyzed_at
            FROM detection_history
            ORDER BY analyzed_at DESC
            LIMIT 20
            """
        )

        records = cursor.fetchall()

        cursor.execute(
            """
            SELECT media_type, COUNT(*)
            FROM detection_history
            GROUP BY media_type
            """
        )

        counts = cursor.fetchall()

        for media_type, count in counts:

            if media_type == "IMAGE":
                image_count = count

            elif media_type == "VIDEO":
                video_count = count

            elif media_type == "AUDIO":
                audio_count = count

            elif media_type == "TEXT":
                text_count = count

        cursor.close()
        connection.close()

    except Exception as e:

        print(
            "DASHBOARD ERROR:"
        )

        print(e)

    return render_template(
        "dashboard.html",
        records=records,
        image_count=image_count,
        video_count=video_count,
        audio_count=audio_count,
        text_count=text_count
    )


# ============================================================
# REPORTS
# ============================================================

@app.route("/reports")
def reports():

    records = []

    total_count = 0
    real_count = 0
    fake_count = 0

    image_count = 0
    video_count = 0
    audio_count = 0
    text_count = 0

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                filename,
                media_type,
                prediction,
                confidence,
                real_probability,
                fake_probability,
                analyzed_at
            FROM detection_history
            ORDER BY analyzed_at DESC
            """
        )

        records = cursor.fetchall()

        total_count = len(
            records
        )

        for record in records:

            prediction = record[3]
            media_type = record[2]

            if prediction == "REAL":
                real_count += 1

            elif prediction == "FAKE":
                fake_count += 1

            if media_type == "IMAGE":
                image_count += 1

            elif media_type == "VIDEO":
                video_count += 1

            elif media_type == "AUDIO":
                audio_count += 1

            elif media_type == "TEXT":
                text_count += 1

        cursor.close()
        connection.close()

    except Exception as e:

        print(
            "REPORTS ERROR:"
        )

        print(e)

    return render_template(
        "reports.html",
        records=records,
        total_count=total_count,
        real_count=real_count,
        fake_count=fake_count,
        image_count=image_count,
        video_count=video_count,
        audio_count=audio_count,
        text_count=text_count
    )


# ============================================================
# ANALYTICS
# ============================================================

@app.route("/analytics")
def analytics():

    records = []

    total_detections = 0

    real_count = 0

    fake_count = 0

    image_count = 0

    video_count = 0

    audio_count = 0

    text_count = 0

    try:

        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                filename,
                media_type,
                prediction,
                confidence,
                real_probability,
                fake_probability,
                analyzed_at
            FROM detection_history
            ORDER BY analyzed_at DESC
            """
        )

        records = cursor.fetchall()

        # ----------------------------------------------------
        # TOTAL DETECTIONS
        # ----------------------------------------------------

        total_detections = len(records)

        # ----------------------------------------------------
        # COUNT DETECTIONS
        # ----------------------------------------------------

        for record in records:

            media_type = record[2]

            prediction = record[3]

            if prediction == "REAL":

                real_count += 1

            elif prediction == "FAKE":

                fake_count += 1

            if media_type == "IMAGE":

                image_count += 1

            elif media_type == "VIDEO":

                video_count += 1

            elif media_type == "AUDIO":

                audio_count += 1

            elif media_type == "TEXT":

                text_count += 1

        cursor.close()

        connection.close()

    except Exception as e:

        print(
            "ANALYTICS ERROR:"
        )

        print(e)

    # --------------------------------------------------------
    # CALCULATE PERCENTAGES
    # --------------------------------------------------------

    real_percentage = 0

    fake_percentage = 0

    if total_detections > 0:

        real_percentage = round(
            (
                real_count
                / total_detections
                * 100
            ),
            2
        )

        fake_percentage = round(
            (
                fake_count
                / total_detections
                * 100
            ),
            2
        )

    # --------------------------------------------------------
    # SEND DATA TO ANALYTICS TEMPLATE
    # --------------------------------------------------------

    return render_template(
        "analytics.html",

        records=records,

        total_detections=total_detections,

        real_count=real_count,

        fake_count=fake_count,

        image_count=image_count,

        video_count=video_count,

        audio_count=audio_count,

        text_count=text_count,

        real_percentage=real_percentage,

        fake_percentage=fake_percentage
    )
# ============================================================
# SETTINGS
# ============================================================

@app.route("/settings")
def settings():

    return render_template(
        "settings.html"
    )


# ============================================================
# MODELS
# ============================================================

@app.route("/models")
def models():

    return render_template(
        "models.html"
    )


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("Synthetic Content Detection Using Machine Learning")
    print("=" * 60)
    print()
    print("Image API  : http://127.0.0.1:5000/predict")
    print("Video API  : http://127.0.0.1:5000/predict-video")
    print("Audio API  : http://127.0.0.1:5000/predict-audio")
    print("Text API   : http://127.0.0.1:5000/predict-text")
    print("Health     : http://127.0.0.1:5000/health")
    print()
    print("Web App    : http://127.0.0.1:5000")
    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )