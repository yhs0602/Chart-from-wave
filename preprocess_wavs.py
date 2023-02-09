import os
import pickle

import librosa
import numpy as np
import torch


def extract_features(wav_file, sr=44100):
    # Load the wav file
    y, sr = librosa.load(wav_file, sr=sr)

    # Pre-processing
    y = librosa.effects.trim(y=y, top_db=60, frame_length=512, hop_length=64)[0]

    # Feature extraction
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Feature normalization
    mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / np.std(
        mfccs, axis=1, keepdims=True
    )

    # Convert to PyTorch tensor
    features = torch.from_numpy(mfccs).float()

    return features


def extract_features_usage():
    # Example usage
    features = extract_features(
        "/Users/yanghyeonseo/gitprojects/dataset20220330/music/song_8037.wav"
    )
    print("Shape of extracted features:", features.shape)


def get_feature_map(dir_path):
    feature_map = {}
    for filename in os.listdir(dir_path):
        if filename.endswith(".wav"):
            print("filename: ", filename)
            file_path = os.path.join(dir_path, filename)
            features = extract_features(file_path)
            feature_map[filename] = features
    return feature_map


def save_feature_map(feature_map, file_path):
    with open(file_path, "wb") as f:
        pickle.dump(feature_map, f)


if __name__ == "__main__":
    dir_path = "/Users/yanghyeonseo/gitprojects/dataset20220330/music"
    feature_map = get_feature_map(dir_path)
    save_feature_map(feature_map, "wave_feature_map.pkl")
    print(feature_map.keys())
