import hashlib
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure

# ---------------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------------
# Number of bins around an amplitude peak that constitute
# its neighborhood. Higher values - less peaks, faster matching
# but potentially worse performance
PEAK_NEIGHB_SIZE = 15

# Number of points to pair the anchor point with to create
# combinatorial hashes. Higher values - more features but
# slower matching
TARGET_ZONE_SIZE = 15

# A percentile of spectrogram magnitude values at which to choose
# a treshold for local peaks
MAGNITUDE_PERCENTILE = 80
# ---------------------------------------------------------------


def get_peaks(spectrogram: np.ndarray) -> zip:
    """
    Calculate local maximums of the spectrogram.

    :param spectrogram: spectrogram of an audio piece;
    :return: zip of frequency and time bin indices of detected local peaks;
    """
    struct = generate_binary_structure(2, 1)
    # dilate the kernel with itself for PEAK_NEIGHB_SIZE iterations;
    # the resulting size of the neighborhood is:
    #   (PEAK_NEIGHB_SIZE * 2 + 1, PEAK_NEIGHB_SIZE * 2 + 1)
    neighborhood = iterate_structure(struct, PEAK_NEIGHB_SIZE)

    # finding a local maximum in each neighborhood
    local_max = maximum_filter(spectrogram, footprint=neighborhood) == spectrogram

    magnitude_thresh = np.percentile(spectrogram, MAGNITUDE_PERCENTILE)

    # leaving only those local maximums that exceed the magnitude threshold
    peak_cond = local_max & (spectrogram > magnitude_thresh)
    freq_idx, time_idx = np.nonzero(peak_cond)

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
    spectrogram = librosa.stft(audio, n_fft=4096, hop_length=2048)
    spectrogram = librosa.amplitude_to_db(np.abs(spectrogram))

    peaks = get_peaks(spectrogram)
    peaks_map = hash_peaks(list(peaks), song_id)

    if show_spec:
        freq_idx, time_idx = zip(*peaks)
        librosa.display.specshow(spectrogram, x_axis='frames', y_axis='frames')
        plt.scatter(time_idx, freq_idx, c='red', s=4)
        plt.show()

    return peaks_map


if __name__ == '__main__':
    audio_path = './data/query_recordings/jazz.00014-snippet-10-10.wav'
    audio, sr = librosa.load(audio_path)
    fingerprint(audio, show_spec=False)
