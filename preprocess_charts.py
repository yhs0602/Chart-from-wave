# data representation: piano roll
# (one-hot encoded start position) : 5 dimensions
# + (one-hot encoded end position) : 5 dimensions
# + (one-hot encoded flick direction) : 5 x 2 dimensions
# + (one-hot-encoded holding group information) : 0: no group, 1: group 1, 2: group 2; 3 x 2 = 6 dimensions
# Total: 5 + 5 + 10 + 6 = 26 dimensions for deresute
import collections
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


def json_to_numpy(data):
    notes = data["notes"]
    bpm = data["metadata"].get("bpm", 120)
    resolution = 4
    features = convert_to_timing_lattice(notes, bpm, resolution)
    return features


def approximate_bpm(timings):
    timings_int = np.round(timings * 1000).astype(int)
    differences = [
        timings_int[i + 1] - timings_int[i] for i in range(timings_int.size - 1)
    ]
    nonzero_differences = [d for d in differences if d > 0]
    intervals = cluster_to_get_bpm(nonzero_differences)
    counter = collections.Counter(intervals)
    approximated_interval = min(counter.most_common(3))
    bpm = 60000 / approximated_interval[0]
    return int(bpm)


def cluster_to_get_bpm(timings):
    from sklearn.cluster import DBSCAN
    import numpy as np

    X = np.array(timings).reshape(-1, 1)

    dbscan = DBSCAN(eps=1.0, min_samples=1)
    clusters = dbscan.fit_predict(X)

    cluster_modes = [
        int(np.mean(X[clusters == i])) for i in np.unique(clusters) if i != -1
    ]
    cluster_modes = np.array(cluster_modes).reshape(-1, 1)

    result = np.zeros_like(X)

    for i, cluster_mode in enumerate(cluster_modes):
        result[np.isin(clusters, i)] = cluster_mode

    return result.flatten().tolist()


# timings = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
# bpm = approximate_bpm(timings)
# print("Approximated BPM:", bpm)

if __name__ == "__main__":
    file_path = "5031___Lunatic Show___MasterPlus"
    with open(file_path, "r") as f:
        data = json.load(f)
    timings = np.array([note["Time"] for note in data["notes"]])
    print(timings)
    bpm = approximate_bpm(timings)
    print(bpm)
