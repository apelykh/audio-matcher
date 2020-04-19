from sklearn.metrics import f1_score
from audio_matcher import AudioMatcher


def evaluate(results: dict):
    """
    Evaluate the performance of AudioMatcher on the folder of audio queries.
    Top-1 accuracy, Top-3 accuracy and F1 score are computed.

    :param results: results obtained from AudioMatcher.match_from_folder();
        a dict of the following structure:
        {
            gt_song_id1: (match_id1, match_id2, match_id3),
            gt_song_id2: (match_id1, match_id2, match_id3),
            ...
        }
    """
    correct_top1 = 0
    correct_top3 = 0

    for gt_song_id, match_ids in results.items():
        if gt_song_id == match_ids[0]:
            correct_top1 += 1

        if gt_song_id in match_ids:
            correct_top3 += 1

    top1_acc = correct_top1 / len(results)
    top3_acc = correct_top3 / len(results)
    f1 = f1_score(y_true=list(results.keys()),
                  y_pred=[match_ids[0] for match_ids in results.values()],
                  average='micro')

    print('-' * 40)
    print('Evaluation results:')
    print('Top-1 acc: {}\nTop-3 acc: {}\nF1 score: {}'.format(top1_acc, top3_acc, f1))


if __name__ == '__main__':
    matcher = AudioMatcher()
    matcher.build_database('./data/database_recordings')
    results = matcher.match_from_folder('./data/query_recordings')
    evaluate(results)
