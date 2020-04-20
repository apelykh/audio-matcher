import librosa
import librosa.display
import numpy as np
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
TARGET_ZONE_SIZE = 50

# A percentile of spectrogram magnitude values at which to choose
# a threshold for local peaks
MAGNITUDE_PERCENTILE = 80
# ---------------------------------------------------------------


def _get_peaks(spectrogram: np.ndarray) -> list:
    """
    Calculate local maximums of the spectrogram.

    :param spectrogram: spectrogram of an audio piece;
    :return: list of tuples with frequency and time bin indices of the
        detected local peaks: [(f1, t1), (f2, t2), ...]
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

    return list(zip(freq_idx, time_idx))


def _hash_peaks(peaks: list, song_id: int) -> dict:
    """
    Apply combinatorial hashing to detected peaks to increase feature entropy.

    :param peaks: list of detected peaks obtained from self.get_peaks();
    :param song_id: a ground-truth song id, used during the database creation.
        During inference phase should be None;
    :return: feature dict of the following structure:
        {
            fingerprint1: (time_offset1, [song_id1]),
            fingerprint2: (time_offset2, [song_id2]),
        }
        Note: song_ids are present only during the DB building phase.
            During inference, values are: (time_offset,)
    """
    # sorting by time indices
    peaks.sort(key=itemgetter(1))
    feature_dict = {}

    # anchor points
    for i in range(len(peaks)):
        # points from target zone
        for j in range(1, TARGET_ZONE_SIZE):
            if i + j >= len(peaks):
                continue
            freq1, time1 = peaks[i]
            freq2, time2 = peaks[i + j]
            time_delta = time2 - time1

            fingerprint = '{}{}{}'.format(str(freq1), str(freq2), str(time_delta))
            # python dict is implemented with hash map ->
            # implicit key hashing while adding element to dict, O(1) access
            feature_dict[fingerprint] = (time1, song_id)\
                if song_id is not None else (time1,)

    return feature_dict


def get_fingerprints(audio: np.array, song_id: int = None, show_spec=False) -> dict:
    """
    Get fingerprints of an audio file.

    :param audio: numpy array of target audio samples;
    :param song_id: a ground-truth song id, used during the database creation.
        During inference phase should be None;
    :param show_spec: if True, a computed spectrogram with local peaks is shown;
    :return: same as in self.hash_peaks();
    """
    spectrogram = librosa.stft(audio, n_fft=4096, hop_length=2048)
    spectrogram = librosa.amplitude_to_db(np.abs(spectrogram))

    peaks = _get_peaks(spectrogram)
    feature_dict = _hash_peaks(peaks, song_id)

    if show_spec:
        freq_idx, time_idx = zip(*peaks)
        librosa.display.specshow(spectrogram, x_axis='frames', y_axis='frames')
        plt.scatter(time_idx, freq_idx, c='blue', s=5)
        plt.show()

    return feature_dict
