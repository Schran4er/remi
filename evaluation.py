import madmom.features.downbeats as downbeats
import numpy as np
import librosa

beats_per_bar = 4


def evaluate_rhythm(audio_path):
    print("path ", audio_path)

    # RNNDownBeatProcessor
    act = downbeats.RNNDownBeatProcessor()(audio_path)
    # DBNDownBeatTrackingProcessor
    proc2 = downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[beats_per_bar, beats_per_bar], fps=100, correct=False)
    # fps=100 because of RNNDownBeatProcessor return,
    # correct=False because this is for evaluation hence aligning the beats to the nearest peak would be wrong
    result = proc2(act)
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
    beat_std = np.std(beat_new[:-1]) # except in this case the last one, as it is longer due to the generation
    print(f"Beat Std = {beat_std} \t \t \t values: {beat_new[:-1]}")

    # Downbeat Std (standard deviation of the bar length differences):
    rawdownbeat_std = [i[0] for i in result]
    downbeat_new = [0] * (len(rawdownbeat_std) // beats_per_bar - 1)
    downbeat_new[0] = rawdownbeat_std[4] - rawdownbeat_std[0]  # because rawdownbeat_std[4] is the time when the first bar ends
    for i in range(1, len(downbeat_new)):
        downbeat_new[i] = rawdownbeat_std[(i + 1) * beats_per_bar] - rawdownbeat_std[i * beats_per_bar]
    downbeat_std = np.std(downbeat_new)
    print(f"Downbeat Std = {downbeat_std} \t \t \t values: {downbeat_new}")

    # Downbeat Salience
    downbeat_salience = [i[1] for i in act]
    salience = np.mean(downbeat_salience)
    print("Downbeat Salience = ", salience)

    return beat_std, downbeat_std, salience


if __name__ == "__main__":
    audio_path = './result_synthesized/initial_from_scratch.wav'
    evaluate_rhythm(audio_path)
