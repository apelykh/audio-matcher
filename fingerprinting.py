import hashlib
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure

# TODO: describe all constants
PEAK_NEIGHBORHOOD_SIZE = 15
# number of points to pair the anchor point with
TARGET_ZONE_SIZE = 15
# HASH_LEN_LIMIT = 20


# TODO: dynamically determine amplitude thresh
def get_peaks(spectrogram: np.ndarray, amplitude_thresh: int = 0,
              show_spec: bool = False) -> zip:
    """

    :param spectrogram:
    :param amplitude_thresh:
    :param show_spec:
    :return:
    """
    struct = generate_binary_structure(2, 1)
    # dilate the kernel with itself for PEAK_NEIGHBORHOOD_SIZE iterations
    # note: does not produce an array of size PEAK_NEIGHBORHOOD_SIZE
    neighborhood = iterate_structure(struct, PEAK_NEIGHBORHOOD_SIZE)

    # finding local maximum in each neighborhood
    local_max = maximum_filter(spectrogram, footprint=neighborhood) == spectrogram

    # leaving only those local maximums that exceed the amplitude threshold
    peak_cond = local_max & (spectrogram > amplitude_thresh)
    freq_idx, time_idx = np.nonzero(peak_cond)

    if show_spec:
        # plt.figure(figsize=(16, 10))
        librosa.display.specshow(spectrogram, x_axis='frames', y_axis='frames')
        plt.scatter(time_idx, freq_idx, c='red', s=4)
        plt.show()

    print('{} peaks generated'.format(len(freq_idx)))

    return zip(freq_idx, time_idx)


def hash_peaks(peaks: list, song_id: int) -> dict:
    # sorting by time indices
    peaks.sort(key=itemgetter(1))

    peaks_map = {}

    # anchor point index
    for i in range(len(peaks)):
        # neighbor point index
        for j in range(1, TARGET_ZONE_SIZE):
            if i + j < len(peaks):
                freq1, time1 = peaks[i]
                freq2, time2 = peaks[i + j]
                time_delta = time2 - time1
                # TODO: introduce empiric constraints on freq/time_delta values?
                hash_source = '{}{}{}'.format(str(freq1), str(freq2), str(time_delta))

                value = (time1, song_id) if song_id is not None else (time1,)
                # python dict is implemented with hash map ->
                # implicit key hashing while adding element to dict
                peaks_map[hash_source] = value

                # h = hashlib.md5(hash_source.encode('utf-8')).hexdigest()
                # yield h[:HASH_LEN_LIMIT], time1
    return peaks_map


def fingerprint(audio: np.array, song_id: int = None, show_spec=False):
    spectrogram = librosa.stft(audio, n_fft=2048, hop_length=1024)
    spectrogram = librosa.amplitude_to_db(np.abs(spectrogram))

    peaks = get_peaks(spectrogram, show_spec=show_spec)
    peaks_map = hash_peaks(list(peaks), song_id)

    return peaks_map


if __name__ == '__main__':
    audio_path = './data/database_recordings/pop.00010.wav'
    audio, sr = librosa.load(audio_path)
    fingerprint(audio)
