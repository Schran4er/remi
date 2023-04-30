import madmom.features.downbeats as downbeats
import numpy as np

beats_per_bar = 4


def evaluate_rhythm(audio_path):
    print("path ", audio_path)
    # RNNDownBeatProcessor
    act = downbeats.RNNDownBeatProcessor()(audio_path)
    # DBNDownBeatTrackingProcessor
    proc2 = downbeats.DBNDownBeatTrackingProcessor(beats_per_bar=[beats_per_bar, beats_per_bar], fps=100)
    result = proc2(act)

    # Beat Std (standard deviation of the beat length differences)
    rawbeat_std = [i[0] for i in result]
    beat_new = [0] * (len(rawbeat_std) - 1)
    beat_new[0] = rawbeat_std[1] - rawbeat_std[0]
    for i in range(1, len(rawbeat_std) - 1):
        beat_new[i] = rawbeat_std[i + 1] - rawbeat_std[i]
    beat_std = np.std(beat_new, ddof=1)
    print("Beat Std = ", beat_std)

    # Downbeat Std (standard deviation of the bar length differences)
    rawdownbeat_std = [i[0] for i in result]
    downbeat_new = [0] * (len(rawdownbeat_std) // beats_per_bar)
    downbeat_new[0] = rawdownbeat_std[4]  # because rawdownbeat_std[4] is the time when the first bar ends
    for i in range(1, len(rawdownbeat_std) // beats_per_bar):
        downbeat_new[i] = rawdownbeat_std[(i + 1) * beats_per_bar - 1] - rawdownbeat_std[i * beats_per_bar - 1]
    downbeat_std = np.std(downbeat_new, ddof=1)
    print("Downbeat Std = ", downbeat_std)

    # Downbeat Salience
    downbeat_salience = [i[1] for i in act]
    salience = sum(downbeat_salience) / len(downbeat_salience)
    print("Downbeat Salience = ", salience)

    downbeat_salience_0 = [i[0] for i in act]
    salience_0 = sum(downbeat_salience_0) / len(downbeat_salience_0)
    print("Downbeat Salience_0 = ", salience_0)

    return beat_std, downbeat_std, salience


if __name__ == "__main__":
    audio_path = './result_synthesized/initial_from_scratch.wav'
    evaluate_rhythm(audio_path)
