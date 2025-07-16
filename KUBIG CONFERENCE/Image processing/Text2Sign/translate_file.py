from models import Translator
from torch.utils.data import Dataset, DataLoader
import torch
import json
from utils import dict_to_device
import os
from datetime import datetime
import sys
from animation_utils import post_translate_parse
from transformers import logging as hf_log


hf_log.set_verbosity_error()


@torch.no_grad()
def iterate_translate(model, data, device, max_length):
    model.eval()
    nias = []
    players = []
    names = []
    print('\nGenerating translations...')
    nmbs = len(data)
    for mb, d in enumerate(data):
        print('Evaluating mb {}/{}'.format(mb + 1, nmbs), end='\r')
        d = dict_to_device(d, device)
        pred = model.batch_translate(
            sequences=d['sentences'],
            max_length=max_length,
            nia_data_script=True,
            player_script=True,
            device=device,
        )
        nias.extend(pred['nia_data_script'])
        players.extend(pred['player_script'])
        names.extend(d['files'])

    return names, nias, players


class JsonTextDataset(Dataset):
    def __init__(self, dir):
        super(JsonTextDataset, self).__init__()
        self.files = os.listdir(dir)
        self.sentences = []
        for file in self.files:
            path = os.path.join(dir, file)
            with open(path, 'rb') as f:
                self.sentences.append(json.load(f)['korean_text'])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return {
            'file': self.files[idx],
            'sentence': self.sentences[idx],
        }

    def collate(self, batch):
        return {
            'files': [b['file'] for b in batch],
            'sentences': [b['sentence'] for b in batch],
        }


def translate(timestamp, path, **params):
    isdir = os.path.isdir(path)

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
    if isdir:
        dataset_test = JsonTextDataset(path)

        dataloader = DataLoader(
            dataset=dataset_test,
            batch_size=params['test']['batch_size'],
            collate_fn=dataset_test.collate,
            shuffle=False,
            num_workers=params['test']['n_dl_workers'],
        )

        names, files_nia, files_player = iterate_translate(
            model,
            data=dataloader,
            device=device,
            max_length=params['max_target_length'],
        )
        files_names = zip(names, files_nia, files_player)

        new_path = os.path.join(
            params['test']['results_path'],
            f'translate_file_{timestamp}/',
        )
        os.mkdir(new_path)

        path_nia = os.path.join(new_path, 'nia/')
        path_player = os.path.join(new_path, 'player/')
        os.mkdir(path_nia)
        os.mkdir(path_player)

        output_save_path = os.path.join(params['test']['results_path'], f'translate_file_{timestamp}/', )
        print(f"\nSaving translations at: {output_save_path}")

        for name, file_nia, file_player in files_names:
            with open(os.path.join(path_nia, name), 'w', encoding='utf8') as f:
                json.dump(file_nia, f, ensure_ascii=False, indent=2)
            with open(os.path.join(path_player, name), 'w', encoding='utf8') as f:
                json.dump(post_translate_parse(file_player), f, ensure_ascii=False, indent=2)
    else:
        with open(path, 'rb') as f:
            sentence = json.load(f)['korean_text']
        name = os.path.split(path)[1]
        file = model.batch_translate(
            sequences=[sentence],
            max_length=params['max_target_length'],
            nia_data_script=True,
            player_script=True,
        )
        file_nia = file['nia_data_script'][0]
        file_player = file['player_script'][0]

        files_names = [(name, file_nia, file_player)]

        path_nia = os.path.join(
            params['test']['results_path'],
            f'translate_file_{timestamp}_nia.json',
        )
        path_player = os.path.join(
            params['test']['results_path'],
            f'translate_file_{timestamp}_player.json',
        )

        output_save_path = os.path.join(params['test']['results_path'], f'translate_file_{timestamp}_*.json', )
        print(f"\nSaving translations at: {output_save_path}")

        for name, file_nia, file_player in files_names:
            with open(path_nia, 'w', encoding='utf8') as f:
                json.dump(file_nia, f, ensure_ascii=False, indent=2)
            with open(path_player, 'w', encoding='utf8') as f:
                json.dump(post_translate_parse(file_player), f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    assert len(sys.argv) > 1, "No input path found."
    path = sys.argv[1]
    print(f'Registered input path: {path}')

    with open('config.json', 'rb') as f:
        parameters = json.load(f)

    translate(timestamp, path, **parameters)
