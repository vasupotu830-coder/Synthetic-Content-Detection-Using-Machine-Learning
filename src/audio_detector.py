import os
import numpy as np
import librosa


def extract_audio_features(audio_path):
    """
    Extract handcrafted audio features from an audio file.

    Returns:
        numpy.ndarray containing the extracted features.
    """

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio
    audio, sample_rate = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    if len(audio) == 0:
        raise ValueError("Audio file contains no samples.")

    # ---------------------------------------------------------
    # 1. Basic signal features
    # ---------------------------------------------------------

    mean_amplitude = np.mean(np.abs(audio))
    amplitude_std = np.std(audio)

    rms = librosa.feature.rms(y=audio)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    zcr_mean = np.mean(zero_crossing_rate)
    zcr_std = np.std(zero_crossing_rate)

    # ---------------------------------------------------------
    # 2. Spectral features
    # ---------------------------------------------------------

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )[0]

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate
    )[0]

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate
    )[0]

    centroid_mean = np.mean(spectral_centroid)
    centroid_std = np.std(spectral_centroid)

    bandwidth_mean = np.mean(spectral_bandwidth)
    bandwidth_std = np.std(spectral_bandwidth)

    rolloff_mean = np.mean(spectral_rolloff)
    rolloff_std = np.std(spectral_rolloff)

    # ---------------------------------------------------------
    # 3. MFCC features
    # ---------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    mfcc_means = np.mean(mfcc, axis=1)

    # ---------------------------------------------------------
    # 4. Combine features
    # ---------------------------------------------------------

    features = [
        mean_amplitude,
        amplitude_std,
        rms_mean,
        rms_std,
        zcr_mean,
        zcr_std,
        centroid_mean,
        centroid_std,
        bandwidth_mean,
        bandwidth_std,
        rolloff_mean,
        rolloff_std
    ]

    # Add the 13 MFCC mean values
    features.extend(mfcc_means.tolist())

    return np.array(features, dtype=np.float32)


if __name__ == "__main__":
    print("Audio feature extraction module loaded successfully.")