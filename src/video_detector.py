import cv2
import os
import joblib
import pandas as pd
import numpy as np

from preprocessing import resize_image
from feature_extraction import extract_features


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "models/deepfake_model_group_split.pkl"


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [
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


# ============================================================
# SETTINGS
# ============================================================

MAX_FRAMES = 20


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# PREDICT SINGLE FRAME
# ============================================================

def predict_frame(frame):

    processed_frame = resize_image(
        frame,
        width=224,
        height=224
    )

    features = extract_features(
        processed_frame
    )

    features_dataframe = pd.DataFrame(
        [features],
        columns=FEATURE_COLUMNS
    )

    prediction = model.predict(
        features_dataframe
    )[0]

    probabilities = model.predict_proba(
        features_dataframe
    )[0]

    fake_probability = probabilities[
        list(model.classes_).index("FAKE")
    ]

    real_probability = probabilities[
        list(model.classes_).index("REAL")
    ]

    return (
        prediction,
        real_probability,
        fake_probability
    )


# ============================================================
# ANALYZE VIDEO
# ============================================================

def analyze_video(video_path):

    if not os.path.exists(video_path):

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )


    video = cv2.VideoCapture(
        video_path
    )


    if not video.isOpened():

        raise ValueError(
            "Could not open the video."
        )


    total_frames = int(
        video.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    fps = video.get(
        cv2.CAP_PROP_FPS
    )


    if total_frames <= 0:

        video.release()

        raise ValueError(
            "Could not read video frames."
        )


    # ========================================================
    # VIDEO DURATION
    # ========================================================

    duration = (
        total_frames / fps
        if fps > 0
        else 0
    )


    # ========================================================
    # FRAME SAMPLING
    # ========================================================

    frame_positions = np.linspace(
        0,
        total_frames - 1,
        min(
            MAX_FRAMES,
            total_frames
        ),
        dtype=int
    )


    results = []


    # ========================================================
    # TERMINAL INFORMATION
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "       VIDEO DEEPFAKE TEST"
    )

    print(
        "=============================="
    )


    print(
        f"\nVideo: {video_path}"
    )

    print(
        f"Total frames: {total_frames}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Duration: {duration:.2f} seconds"
    )

    print(
        f"Frames analyzed: {len(frame_positions)}"
    )


    print(
        "\nAnalyzing frames...\n"
    )


    # ========================================================
    # ANALYZE SELECTED FRAMES
    # ========================================================

    for index, frame_number in enumerate(
        frame_positions,
        start=1
    ):

        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_number)
        )


        success, frame = video.read()


        if not success:

            print(
                f"Frame {index}: Could not read"
            )

            continue


        try:

            (
                prediction,
                real_probability,
                fake_probability
            ) = predict_frame(frame)


            results.append({

                "prediction": prediction,

                "real_probability":
                    real_probability,

                "fake_probability":
                    fake_probability

            })


            print(

                f"Frame {index:02d} | "

                f"Prediction: {prediction} | "

                f"REAL: "
                f"{real_probability * 100:.2f}% | "

                f"FAKE: "
                f"{fake_probability * 100:.2f}%"

            )


        except Exception as error:

            print(
                f"Frame {index}: Error: {error}"
            )


    video.release()


    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if not results:

        raise ValueError(
            "No frames could be analyzed."
        )


    # ========================================================
    # AVERAGE PROBABILITIES
    # ========================================================

    average_real = np.mean([

        result["real_probability"]

        for result in results

    ])


    average_fake = np.mean([

        result["fake_probability"]

        for result in results

    ])


    # ========================================================
    # FINAL PREDICTION
    # ========================================================

    if average_fake > average_real:

        final_prediction = "FAKE"

    else:

        final_prediction = "REAL"


    confidence = max(
        average_real,
        average_fake
    )


    # ========================================================
    # FRAME COUNTS
    # ========================================================

    fake_frames = sum(

        result["prediction"] == "FAKE"

        for result in results

    )


    real_frames = sum(

        result["prediction"] == "REAL"

        for result in results

    )


    # ========================================================
    # TERMINAL RESULT
    # ========================================================

    print(
        "\n=============================="
    )

    print(
        "        VIDEO RESULT"
    )

    print(
        "=============================="
    )


    print(
        f"\nVideo duration: "
        f"{duration:.2f} seconds"
    )


    print(
        f"Frames analyzed: "
        f"{len(results)}"
    )


    print(
        f"REAL frame predictions: "
        f"{real_frames}"
    )


    print(
        f"FAKE frame predictions: "
        f"{fake_frames}"
    )


    print(
        f"\nAverage REAL probability: "
        f"{average_real * 100:.2f}%"
    )


    print(
        f"Average FAKE probability: "
        f"{average_fake * 100:.2f}%"
    )


    print(
        f"\nFinal Prediction: "
        f"{final_prediction}"
    )


    print(
        f"Final Confidence: "
        f"{confidence * 100:.2f}%"
    )


    print(
        "\n=============================="
    )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "prediction":
            final_prediction,

        "confidence":
            confidence * 100,

        "real_probability":
            average_real * 100,

        "fake_probability":
            average_fake * 100,

        "frames_analyzed":
            len(results),

        "real_frames":
            real_frames,

        "fake_frames":
            fake_frames,

        "duration":
            duration

    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    video_path = input(
        "\nEnter video path: "
    ).strip()


    try:

        analyze_video(
            video_path
        )


    except Exception as error:

        print(
            f"\nError: {error}"
        )