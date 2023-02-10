import os
import pickle

import librosa
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


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


def save_feature_map_as_numpy(feature_map, file_path):
    np.save(file_path, feature_map)


# if __name__ == "__main__":
#     dir_path = "/Users/yanghyeonseo/gitprojects/dataset20220330/music"
#     feature_map = get_feature_map(dir_path)
#     save_feature_map(feature_map, "wave_feature_map.pkl")
#     print(feature_map.keys())


def pad_sequences():
    # Load the pickled file
    with open("wave_feature_map.pkl", "rb") as f:
        data = pickle.load(f)
    for key, value in data.items():
        print(key, value.shape)
    # Pad the sequences
    padded_sequences = pad_sequence(
        [value.T for value in data.values()], batch_first=True, padding_value=0
    )
    print(padded_sequences.shape, type(padded_sequences))
    for key, value in zip(data.keys(), padded_sequences):
        print(key, value.shape)
    list_padded_sequences = list(padded_sequences)
    padded_data = dict(
        {key: value for key, value in zip(data.keys(), list_padded_sequences)}
    )
    dieted_data = {key: torch.tensor(value.T) for key, value in padded_data.items()}
    save_feature_map(dieted_data, "padded_wave_feature_map.pkl")


if __name__ == "__main__":
    pass
    # pad_sequences()
