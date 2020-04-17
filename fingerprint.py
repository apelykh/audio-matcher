import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, iterate_structure

PEAK_NEIGHBORHOOD_SIZE = 15


# TODO: dynamically determine amplitude thresh
def get_peaks(spectrogram: np.ndarray, amplitude_thresh: int = 0,
              show_spec: bool = False):
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

    # amplitudes = spectrogram[peak_cond]

    if show_spec:
        # plt.figure(figsize=(16, 10))
        librosa.display.specshow(spectrogram, x_axis='frames', y_axis='frames')
        plt.scatter(time_idx, freq_idx, c='red', s=4)
        plt.show()

    return zip(freq_idx, time_idx)


def fingerprint(audio: np.array):
    spectrogram = librosa.stft(audio, n_fft=2048, hop_length=1024)
    spectrogram = librosa.amplitude_to_db(np.abs(spectrogram))

    peaks = get_peaks(spectrogram, show_spec=True)


if __name__ == '__main__':
    audio_path = './data/database_recordings/classical.00000.wav'
    audio, sr = librosa.load(audio_path)
    fingerprint(audio)
