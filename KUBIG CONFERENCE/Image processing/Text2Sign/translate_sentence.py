from models import Translator
import torch
import json
import os
import sys
from datetime import datetime
from animation_utils import post_translate_parse


def translate(timestamp, sentence, **params):
    with open(params['ksl_vocab_path'], 'rb') as f:
        vocab = json.load(f)

    sign_vocab = list(set(vocab['sign_vocab']))
    nms_vocab = list(set(vocab['nms_vocab']))
    device = params['test']['device']

    model = Translator(
        sign_vocab=sign_vocab,
        nms_vocab=nms_vocab,
        ksl_embedding_method=params['ksl_embedding_method'],
        ksl_embedding_split=params['ksl_embedding_split'],
        animation_mapping_path=params['animation_mapping_path'],
        device=device,
    ).to(device)
    print('\nLoading model weights...')
    if params['test']['load_model'] and params['test']['load_id'] != '':
        model.load_state_dict(torch.load(
            '{}{}_model.pt'.format(params['test']['model_path'], params['test']['load_id']),
            map_location=device
        ))

    model.eval()

    print('\nTranslating...')
    output = model.batch_translate(
        sequences=[sentence],
        max_length=params['max_target_length'],
        nia_data_script=True,
        player_script=True,
    )
    output_nia = output['nia_data_script'][0]
    output_player = post_translate_parse(output['player_script'][0])

    path_nia = os.path.join(
        params['test']['results_path'],
        f'translate_sentence_{timestamp}_nia.json',
    )
    path_player = os.path.join(
        params['test']['results_path'],
        f'translate_sentence_{timestamp}_player.json',
    )

    output_save_path = os.path.join(params['test']['results_path'], f'translate_sentence_{timestamp}_*.json',)
    print(f"\nSaving translations at: {output_save_path}")

    with open(path_nia, 'w', encoding='utf8') as f:
        json.dump(output_nia, f, ensure_ascii=False, indent=2)
    with open(path_player, 'w', encoding='utf8') as f:
        json.dump(output_player, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    assert len(sys.argv) > 1, "No input sentence found."
    sentence = sys.argv[1]
    print(f'Registered input: {sentence}')
    if len(sentence.split(' ')) == 1:
        print('Input is only one word long')
        print('If translating a multiple-word sentence, please put the whole sentence in quotes.')

    with open('config.json', 'rb') as f:
        parameters = json.load(f)
    translate(timestamp, sentence, **parameters)
