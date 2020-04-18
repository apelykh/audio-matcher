# To aid in automatic testing, your code must be callable in two parts as follows:
#     fingerprintBuilder(/path/to/database/, /path/to/fingerprints/)
#     audioIdentification(/path/to/queryset/, /path/to/fingerprints/, /path/to/output.txt)
#
# The format of the output.txt file will include one line for each query audio recording, using the
# following format:
# query audio 1.wav database audio x1.wav database audio x2.wav database audio x3.wav
# query audio 2.wav database audio y1.wav database audio y2.wav database audio y3.wav
# ...
# where the filenames in each line are separated by a tab, and the three database recordings per line are
# the top three ones which are identified by the audio identification system as more closely matching
# the query recording, ranked from first to third.

import os
import time
import pickle
import librosa
from fingerprinting import fingerprint


class AudioMatcher:
    def __init__(self):
        self.id_to_song = {}
        self.fingerprints_db = {}

    def build_database(self, audio_folder_path, pickle_path=None):
        if pickle_path:
            # load the database from file
            with open(pickle_path, 'r') as f:
                self.fingerprints_db = pickle.load(f)
        else:
            # build the database
            for i, audio_file in enumerate(os.listdir(audio_folder_path)):
                print(audio_file)
                self.id_to_song[i] = audio_file

                audio_path = os.path.join(audio_folder_path, audio_file)
                s1 = time.time()
                audio, sr = librosa.load(audio_path)
                read_time = time.time() - s1

                s2 = time.time()
                song_peaks_map = fingerprint(audio, song_id=i)
                finger_time = time.time() - s2
                print("--- read: {}, fingerprint: {} ---".format(read_time, finger_time))

                self.fingerprints_db.update(song_peaks_map)

                if i == 20:
                    break

        print('DB contains {} fingerprints'.format(len(self.fingerprints_db)))

    def match_song(self, song_path):
        audio, sr = librosa.load(song_path)
        song_peaks_map = fingerprint(audio, show_spec=True)

        # mapping song ids to db-query time offset pairs
        id_to_time_pairs = {}

        s1 = time.time()
        for feature in song_peaks_map:
            if feature in self.fingerprints_db:
                db_song_offset, song_id = self.fingerprints_db[feature]
                query_song_offset = song_peaks_map[feature][0]

                if song_id not in id_to_time_pairs:
                    id_to_time_pairs[song_id] = []
                id_to_time_pairs[song_id].append((db_song_offset, query_song_offset))

        search_time = time.time() - s1
        print('Search time: ', search_time)

        for k, v in id_to_time_pairs.items():
            print(k, len(v))


if __name__ == '__main__':
    matcher = AudioMatcher()
    matcher.build_database('./data/database_recordings')
    print('-' * 40)
    matcher.match_song('./data/query_recordings/pop.00049-snippet-10-20.wav')
