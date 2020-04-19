import os
import pickle
import librosa
import numpy as np
from fingerprinting import fingerprint


class AudioMatcher:
    def __init__(self):
        self.id_to_song = {}
        self.song_to_id = {}
        self.fingerprints_db = {}

    def _build_song_mappings(self, audio_folder_path: str):
        """
        Build self.id_to_song mapping when the database is loaded from file.

        :param audio_folder_path:
        """
        for i, audio_file in enumerate(sorted(os.listdir(audio_folder_path))):
            self.id_to_song[i] = audio_file
        self.song_to_id = {v: k for k, v in self.id_to_song.items()}

    def build_database(self, audio_folder_path: str, db_filepath: str = './data/fingerprints_db.pkl'):
        if os.path.isfile(db_filepath):
            print('[.] Loading the databse...')

            with open(db_filepath, 'rb') as f:
                self.fingerprints_db = pickle.load(f)
            self._build_song_mappings(audio_folder_path)
        else:
            print('[.] Building the databse...')

            # sort the files to ensure consistent order
            listdir = sorted(os.listdir(audio_folder_path))
            for i, audio_file in enumerate(listdir):
                if not audio_file[-4:] == '.wav':
                    continue

                if i % 30 == 0:
                    print('File {}/{}'.format(i, len(os.listdir(audio_folder_path))))
                self.id_to_song[i] = audio_file
                self.song_to_id[audio_file] = i

                audio_path = os.path.join(audio_folder_path, audio_file)
                audio, sr = librosa.load(audio_path)
                song_peaks_map = fingerprint(audio, song_id=i)
                self.fingerprints_db.update(song_peaks_map)

            with open(db_filepath, 'wb') as f:
                pickle.dump(self.fingerprints_db, f)

        print('DB contains {} fingerprints'.format(len(self.fingerprints_db)))

    def match_song(self, song_path: str, num_results=3) -> tuple:
        audio, sr = librosa.load(song_path)
        song_peaks_map = fingerprint(audio)

        # mapping song ids to db-query time offset pairs
        id_to_time_diffs = {}

        for feature in song_peaks_map:
            if feature in self.fingerprints_db:
                db_song_offset, song_id = self.fingerprints_db[feature]
                query_song_offset = song_peaks_map[feature][0]
                time_diff = db_song_offset - query_song_offset

                if song_id not in id_to_time_diffs:
                    id_to_time_diffs[song_id] = []
                id_to_time_diffs[song_id].append(time_diff)

        # count the frequencies of unique time differences for each song
        song_ids = []
        max_counts = []
        for song_id, time_diffs in id_to_time_diffs.items():
            _, counts = np.unique(time_diffs, return_counts=True)
            max_counts.append(int(np.max(counts)))
            song_ids.append(song_id)

        res_indices = np.argsort(max_counts)[::-1][:num_results]

        return tuple(song_ids[i] for i in res_indices)

    def match_from_folder(self, queries_folder: str, output_filepath='./results.txt') -> dict:
        results = {}

        if output_filepath:
            f = open(output_filepath, 'w')

        print('[.] Matching songs...')
        listdir = sorted(os.listdir(queries_folder))
        for i, audio_file in enumerate(listdir):
            if not audio_file[-4:] == '.wav':
                continue

            if i % 30 == 0:
                print('File {}/{}'.format(i, len(listdir)))

            gt_song_name = '{}.wav'.format(audio_file.split('-')[0])
            gt_song_id = self.song_to_id[gt_song_name]

            audio_path = os.path.join(queries_folder, audio_file)
            matched_ids = self.match_song(audio_path)
            results[gt_song_id] = matched_ids

            if output_filepath:
                song_names = [self.id_to_song[song_id] for song_id in matched_ids]
                res_string = '{}\t{}\t{}\t{}\n'.format(gt_song_name, *song_names)
                f.write(res_string)

        if output_filepath:
            f.close()

        return results
