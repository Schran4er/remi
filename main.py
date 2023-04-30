from model import PopMusicTransformer
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# if no prompt given: generate from scratch
#  otherwise: generate continuation
def generate(model: PopMusicTransformer, output_path: str, prompt=None):
    model.generate(
        n_target_bar=16,
        temperature=1.2,
        topk=5,
        output_path=output_path,
        prompt=prompt)


def main():
    # declare model
    model = PopMusicTransformer(
        checkpoint='REMI-tempo-checkpoint',
        # checkpoint = 'REMI-tempo-chord-checkpoint',
        # checkpoint='my_midi_data_checkpoint',
        is_training=False)

    # generate from scratch
    for i in range(0, 10):
        output_path = f'./result/tempo_checkpoint_from_scratch_{i}.midi'
        generate(model, output_path)

    # generate continuation
    # output_path = './result/continuation.midi'
    # prompt = './data/evaluation/000.midi'
    # generate(model, output_path, prompt)

    # close model
    model.close()


if __name__ == '__main__':
    main()
