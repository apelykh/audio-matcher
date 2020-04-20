import os
import time
import pickle
import librosa
import numpy as np
from fingerprinting import get_fingerprints


class AudioMatcher:
    def __init__(self, verbose=True):
        """
        :param verbose: if True, console logs will be displayed;
        """
        self.id_to_song = {}
        self.song_to_id = {}
        self.fingerprints_db = {}
        self.verbose = verbose

    def _build_song_mappings(self, audio_folder_path: str):
        """
        Build id-to-song and song-to-id mappings when the database is loaded from file.

        :param audio_folder_path: path to the database audio files;
        """
        for i, audio_file in enumerate(sorted(os.listdir(audio_folder_path))):
            self.id_to_song[i] = audio_file
        self.song_to_id = {v: k for k, v in self.id_to_song.items()}

    def _build_database(self, audio_folder_path: str):
        """
        Build the audio fingerprints database from files;

        :param audio_folder_path: path to the database audio files;
        """
        # sort the files to ensure consistent order
        listdir = sorted(os.listdir(audio_folder_path))

        for i, audio_file in enumerate(listdir):
            if not audio_file[-4:] == '.wav':
                continue
            if i % 30 == 0 and self.verbose:
                print('File {}/{}'.format(i, len(os.listdir(audio_folder_path))))
            self.id_to_song[i] = audio_file
            self.song_to_id[audio_file] = i

            audio_path = os.path.join(audio_folder_path, audio_file)
            audio, sr = librosa.load(audio_path)
            feature_dict = get_fingerprints(audio, song_id=i)
            # extend the db with new features for the current audio file
            self.fingerprints_db.update(feature_dict)

    def get_database(self, audio_folder_path: str, db_filepath: str = './data/fingerprints_db.pkl'):
        """
        Build the audio fingerprints database or load it from file.

        :param audio_folder_path: path to the database audio files;
        :param db_filepath: path to the database file. If file exists, it will be loaded,
            otherwise - the db will be computed and saved to the specified location;
            Note: user is responsible for existence of all the sub-folders in the path;
        """
        if os.path.isfile(db_filepath):
            if self.verbose:
                print('[.] Loading the databse...')

            with open(db_filepath, 'rb') as f:
                self.fingerprints_db = pickle.load(f)
            self._build_song_mappings(audio_folder_path)
        else:
            if self.verbose:
                print('[.] Building the databse...')
            t1 = time.time()
            self._build_database(audio_folder_path)
            if self.verbose:
                print('[+] DB built in {} sec.'.format(time.time() - t1))

            with open(db_filepath, 'wb') as f:
                pickle.dump(self.fingerprints_db, f)

        if self.verbose:
            print('DB contains {} fingerprints\n'.format(len(self.fingerprints_db)))

    @staticmethod
    def _decode_time_diffs(id_to_time_diffs: dict, num_results: int) -> tuple:
        """
        Decode time differences between db and query features and check their
        temporal alignment.

        :param id_to_time_diffs: mapping of song ids to db-query time offset
            differences;
        :param num_results: number of matches returned;
        :return: a tuple of size num_results with matched song indices;
        """
        song_ids = []
        max_counts = []
        # count the frequencies of unique time differences for each song
        for song_id, time_diffs in id_to_time_diffs.items():
            _, counts = np.unique(time_diffs, return_counts=True)
            max_counts.append(int(np.max(counts)))
            song_ids.append(song_id)

        # constant time difference between db and query features indicates
        # their same relative offset -> most likely a match

        # indices of songs with maximum number of constant offset differences
        res_indices = np.argsort(max_counts)[::-1][:num_results]

        return tuple(song_ids[i] for i in res_indices)

    def match_song(self, song_path: str, num_results: int = 3) -> tuple:
        """
        Match a query audio file with a database.

        :param song_path: path to the query song;
        :param num_results: number of matches returned;
        :return: a tuple of size num_results with matched song indices;
            Note: indices can be decoded to song names with self.id_to_song;
        """
        audio, sr = librosa.load(song_path)
        feature_dict = get_fingerprints(audio)

        # mapping song ids to db-query time offset differences
        id_to_time_diffs = {}

        for fingerprint in feature_dict:
            # if the feature match is found
            if fingerprint in self.fingerprints_db:
                db_song_offset, song_id = self.fingerprints_db[fingerprint]
                query_song_offset = feature_dict[fingerprint][0]
                time_diff = db_song_offset - query_song_offset

                if song_id not in id_to_time_diffs:
                    id_to_time_diffs[song_id] = []
                id_to_time_diffs[song_id].append(time_diff)

        song_matches = self._decode_time_diffs(id_to_time_diffs, num_results)

        return song_matches

    def match_from_folder(self, queries_folder: str, num_results: int = 3,
                          output_filepath='./results.txt') -> dict:
        """
        Identify all query audio files from a given folder.

        :param queries_folder: path to the folder with query songs;
        :param num_results: number of matches returned;
        :param output_filepath: path to the file where results will be saved
            using the following format:
                query_1.wav db_song_x1.wav  db_song_x2.wav  db_song_x3.wav ...
                query_2.wav db_song_y1.wav  db_song_y2.wav  db_song_y3.wav ...
                ...
            If None, results will not be saved to a file;
        :return: result dict of a following format:
            {
                gt_song_id1: (match_1, ..., match_num_results),
                gt_song_id2: (match_1, ..., match_num_results),
                ...
            }
            Note: returned dict is used for evaluation;
        """
        results = {}
        if output_filepath:
            f = open(output_filepath, 'w')

        if self.verbose:
            print('[.] Matching songs...')

        listdir = sorted(os.listdir(queries_folder))
        for i, audio_file in enumerate(listdir):
            if not audio_file[-4:] == '.wav':
                continue

            if i % 30 == 0 and self.verbose:
                print('query {}/{}'.format(i, len(listdir)))

            audio_path = os.path.join(queries_folder, audio_file)
            matched_ids = self.match_song(audio_path, num_results)
            gt_song_name = '{}.wav'.format(audio_file.split('-')[0])
            gt_song_id = self.song_to_id[gt_song_name]
            results[gt_song_id] = matched_ids

            if output_filepath:
                song_names = [self.id_to_song[song_id] for song_id in matched_ids]
                res_string = '{}\t{}\t{}\t{}\n'.format(gt_song_name, *song_names)
                f.write(res_string)

        if output_filepath:
            f.close()

        return results
