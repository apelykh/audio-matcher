# A janky version of Shazam

## Description
The proposed algorithm uses a combinatorially hashed time-frequency constellation analysis of
audio file and based on the paper of Wang [1].

On a high level, the algorithm relies on fingerprinting each audio file based on its spectral representation and storing the features of “database” tracks. The fingerprints of “query” files are subsequently matched against the database and the correctness of match is verified by checking a temporal correspondance of features.


## Usage
```python
from audio_matcher import AudioMatcher

matcher = AudioMatcher()
# compute features from audio files
matcher.get_database(PATH_TO_DB_FOLDER)
# match individual song
song_matches = matcher.match_song(SONG_PATH)
# run on the folder with query songs
results = matcher.match_from_folder(PATH_TO_QUERIES_FOLDER)
# evaluate the metrics
evaluate(results)
```

## Evaluation
The system's performance was measured on a subset of GTZAN dataset that contains 300 classical, jazz, and pop
audio clips. The data can be downloaded from here: [database tracks](https://drive.google.com/open?id=1XXeRXCf295gSJz80-IU1DOX4RKj7iRLX), [query tracks](https://drive.google.com/open?id=1oKP-hOsC945MyuH5iajSdlPMkIMLNeIe)

 metric | classical* | jazz* | pop* | classical | jazz | pop | overall
-- | ------------------- | -------------- | ------------- | --------- | ---- | --- | -------
Top-1 accuracy | 0.649 | 0.622 | 0.892 | 0.703 | 0.649 | 0.892 | **0.721**
Top-3 accuracy | 0.703 | 0.676 | 0.973 | 0.703 | 0.676 | 0.973 | **0.784**
F1 score | 0.649 | 0.622 | 0.892 | 0.703 | 0.649 | 0.892 | **0.721**

For the experiments marked with __*__, the full-size database was used but queries were limited to an indicated genre. 


## References
[1] A. Wang, “An Industrial Strength Audio Search Algorithm,” in Proc. of the 4th Intl. Society for
Music Information Retrieval Conf. (ISMIR), 2003.
