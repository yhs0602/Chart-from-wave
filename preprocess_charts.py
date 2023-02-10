# data representation: piano roll
# (one-hot encoded start position) : 5 dimensions
# + (one-hot encoded end position) : 5 dimensions
# + (one-hot encoded flick direction) : 5 x 2 dimensions
# + (one-hot-encoded holding group information) : 0: no group, 1: group 1, 2: group 2; 3 x 2 = 6 dimensions
# Total: 5 + 5 + 10 + 6 = 26 dimensions for deresute

import json

import numpy as np


def convert_to_timing_lattice(notes, bpm, resolution):
    """Convert note information to timing lattice representation.

    Args:
        notes: List of notes, each note should have information on start position, end position, flick direction, and holding group information.
        bpm: Beats per minute of the chart.
        resolution: Number of time intervals to divide each note into.

    Returns:
        numpy array of shape (len(notes), 26, resolution), where the last dimension is the timing lattice representation of the notes.
    """
    interval_duration = 60 / bpm / resolution
    note_vectors = []
    for note in notes:
        start = note["Time"]
        start_line = int(note["StartLine"])
        end_line = int(note["EndLine"])
        flick = note["Flick"]
        mode = note["Mode"]
        # group = note["Group"]
        group = 1

        # Encode start position, end position, flick direction, and holding group information
        start_one_hot = np.zeros(5)
        start_one_hot[start_line - 1] = 1
        end_one_hot = np.zeros(5)
        end_one_hot[end_line - 1] = 1
        flick_one_hot = np.zeros(5 * 2)
        flick_one_hot[flick * 2] = 1
        flick_one_hot[flick * 2 + 1] = 1
        group_one_hot = np.zeros(3 * 2)
        group_one_hot[group * 2] = 1
        group_one_hot[group * 2 + 1] = 1

        note_vector = np.concatenate(
            [start_one_hot, end_one_hot, flick_one_hot, group_one_hot]
        )

        note_vectors.append(note_vector)
        # Fill in the gaps with zeros until the next note
        for i in range(1, resolution):
            note_vectors.append(np.zeros(26))

    return np.array(note_vectors)


def json_to_numpy(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    notes = data["notes"]
    bpm = data["metadata"].get("bpm", 120)
    resolution = 4
    features = convert_to_timing_lattice(notes, bpm, resolution)
    return features


if __name__ == "__main__":
    file_path = "/Users/yanghyeonseo/gitprojects/dataset20220330/twFiles7/5031___Lunatic Show___MasterPlus"
    features = json_to_numpy(file_path)
    print(features.shape)
