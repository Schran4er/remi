import madmom.features.downbeats as downbeats
import numpy as np
import librosa
import glob
import warnings
from numpy import VisibleDeprecationWarning
import pandas as pd

beats_per_bar = 4


def evaluate_rhythm(audio_path):
    # RNNDownBeatProcessor
    act = downbeats.RNNDownBeatProcessor()(audio_path)
    # DBNDownBeatTrackingProcessor
    processor = downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[beats_per_bar, beats_per_bar], fps=100,
                                                       correct=True)
    # fps=100 because of RNNDownBeatProcessor return,
    # correct=False because this is for evaluation hence aligning the beats to the nearest peak would be wrong
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)  # ignore warning caused by madmom
        result = processor(act)
    # length of the track (required later)
    audio_length = librosa.get_duration(filename=audio_path)

    # Beat Std (standard deviation of the beat length differences):
    rawbeat_std = [i[0] for i in result]
    beat_new = [0] * len(rawbeat_std)

    # set length of last element
    last_element_index = len(rawbeat_std) - 1
    beat_new[last_element_index] = audio_length - rawbeat_std[last_element_index]

    for i in range(0, len(rawbeat_std) - 1):
        beat_new[i] = rawbeat_std[i + 1] - rawbeat_std[i]
    beat_std = np.std(beat_new[:-1])  # except in this case the last one, as it is longer due to the generation

    # Downbeat Std (standard deviation of the bar length differences):
    rawdownbeat_std = [i[0] for i in result]
    downbeat_new = [0] * (len(rawdownbeat_std) // beats_per_bar - 1)
    downbeat_new[0] = rawdownbeat_std[4] - rawdownbeat_std[
        0]  # because rawdownbeat_std[4] is the time when the first bar ends
    for i in range(1, len(downbeat_new)):
        downbeat_new[i] = rawdownbeat_std[(i + 1) * beats_per_bar] - rawdownbeat_std[i * beats_per_bar]
    downbeat_std = np.std(downbeat_new)

    # Downbeat Salience
    downbeat_salience = [i[1] for i in act]
    salience = np.mean(downbeat_salience)

    return beat_std, beat_new[:-1], downbeat_std, downbeat_new, salience


if __name__ == "__main__":
    # TODO: implement "time taken"
    print("start")
    # audio_folder = './result_synthesized/'  # input folder
    audio_folder = 'C:/Users/Alex/AppData/Roaming/REAPER/ProjectTemplates/result_synthesized/'
    all_results = [[], [], []]
    all_results_length = 0

    results = []
    counter = 0
    for audio_path in glob.glob(f'{audio_folder}*.wav'):
        if counter % 50 == 0:
            df = pd.DataFrame.from_records(results, columns=['beat_std', 'downbeat_std', 'salience'])
            df.to_csv(f'synthesized_eval_results_{counter}.csv')

        beat_std, beat_new, downbeat_std, downbeat_new, salience = evaluate_rhythm(audio_path)
        # prints:
        print(f"path: {audio_path}")
        print(f"Beat Std = \t \t \t{beat_std}")
        print(f"Downbeat Std = \t \t{downbeat_std}")
        print(f"Downbeat Salience = {salience}\n")

        # all_results[0].append(beat_std)
        # all_results[1].append(downbeat_std)
        # all_results[2].append(salience)
        results.append((beat_std, downbeat_std, salience))
        counter += 1

    df = pd.DataFrame.from_records(results, columns=['beat_std', 'downbeat_std', 'salience'])
    df.to_csv('synthesized_eval_results.csv')

    # calculate mean result values
    # result_beat_std = np.mean(all_results[0])
    # result_downbeat_std = np.mean(all_results[1])
    # result_salience = np.mean(all_results[2])
    # print(f"all_results: (mean)\nBeat Std = \t \t \t{result_beat_std}\nDownbeat Std = \t \t{result_downbeat_std}\nDownbeat "
    #       f"salience = {result_salience}")
