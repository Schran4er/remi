from model import PopMusicTransformer
from glob import glob
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    # declare model
    model = PopMusicTransformer(
        checkpoint='REMI-tempo-checkpoint',
        is_training=True)
    # prepare data
    midi_paths = glob('my_midi_data/*.midi')  # TODO: replace with path to MIDI data folder to train on from checkpoint
    training_data = model.prepare_data(midi_paths=midi_paths)

    # check output checkpoint folder
    ####################################
    # if you use "REMI-tempo-chord-checkpoint" for the pre-trained checkpoint
    # please name your output folder as something with "chord"
    # for example: my-love-chord, cute-doggy-chord, ...
    # if use "REMI-tempo-checkpoint"
    # for example: my-love, cute-doggy, ...
    ####################################
    # output_checkpoint_folder = 'REMI-finetune'  # your decision
    output_checkpoint_folder = 'my_midi_data_finetune'
    if not os.path.exists(output_checkpoint_folder):
        os.mkdir(output_checkpoint_folder)

    # finetune
    model.finetune(
        training_data=training_data,
        output_checkpoint_folder=output_checkpoint_folder)

    ####################################
    # after finetuning, please choose which checkpoint you want to try
    # and change the checkpoint names you choose into "model"
    # and copy the "dictionary.pkl" into your output_checkpoint_folder
    # ***** the same as the content format in "REMI-tempo-checkpoint" *****
    # and then, you can use "main.py" to generate your own music!
    # (do not forget to revise the checkpoint path to your own in "main.py")
    ####################################

    # close
    model.close()


if __name__ == '__main__':
    main()

