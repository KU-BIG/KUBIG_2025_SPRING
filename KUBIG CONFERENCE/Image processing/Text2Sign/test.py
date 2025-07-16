from models import Translator
from data_utils import BLEUDataset
from torch.utils.data import DataLoader
import torch
import json
from utils import bleu_test, seed_all
import pickle
import os
import sys


def main():
    with open('config.json', 'rb') as f:
        params = json.load(f)

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
        channel_to_gloss=params['channel_to_gloss'],
        animation_mapping_path=params['animation_mapping_path'],
        device=device,
    ).to(device)

    if params['test']['load_model'] and params['test']['load_id'] != '':
        print('\nLoading model weights...')
        model.load_state_dict(torch.load(
            '{}{}_best_model.pt'.format(params['test']['model_path'], params['test']['load_id']),
            map_location=device
        ))

    model.eval()

    if params['use_dataset_pickles']:
        bleu_test_pickle_path = os.path.join(params['pickle_directory'], params['bleu_test_pickle_name'])
        with open(bleu_test_pickle_path, 'rb') as f:
            dataset_test = pickle.load(f)
    else:
        dataset_test = BLEUDataset(path='data/test/', model=model)
    dataloader = DataLoader(
        dataset=dataset_test,
        batch_size=params['test']['batch_size'],
        collate_fn=dataset_test.collate,
        shuffle=False,
        num_workers=params['test']['n_dl_workers'],
    )

    # Always zero and takes too long to calculate with untrained model -> just set to zero and note in README
    bleu_score_pre = 0

    bleu_score_post = bleu_test(
        model=model,
        data=dataloader,
        device=params['test']['device'],
        n_beams=params['test']['n_beams'],
        top_k=params['test']['top_k'],
        top_p=params['test']['top_p'],
        max_length=params['max_target_length'],
        ngram=4,
    )

    print('\n\nBLEU score pre train: {}\nBLEU score post train: {}'.format(bleu_score_pre, bleu_score_post))


if __name__ == '__main__':
    # set seed
    args = sys.argv
    try:
        seed = int(args[1])
        given_seed = True
    except:
        seed = torch.randint(0, 100_000, (1,))[0].item()
        given_seed = False
    seed_all(seed)
    print('Using {} seed {}\n'.format(
        'given' if given_seed else 'random',
        seed,
    ))

    # run test
    main()
