from models import Translator
import json
from data_utils import JsonDataset, BLEUDataset
import pickle
import os


def pickle_it(d, path):
    with open(path, 'wb') as f:
        pickle.dump(d, f)


def main(**kwargs):
    with open(kwargs['ksl_vocab_path'], 'rb') as f:
        vocab = json.load(f)

    sign_vocab = list(set(vocab['sign_vocab']))
    nms_vocab = list(set(vocab['nms_vocab']))
    model = Translator(
        sign_vocab=sign_vocab,
        nms_vocab=nms_vocab,
        max_source_length=kwargs['max_source_length'],
        max_target_length=kwargs['max_target_length'],
        animation_mapping_path=kwargs['animation_mapping_path'],
        ignore_nms=kwargs['ignore_nms'],
        device='cpu',
    ).to('cpu')

    process_train = os.path.isdir(kwargs['data_train_path'])
    process_validate = os.path.isdir(kwargs['data_validate_path'])
    process_test = os.path.isdir(kwargs['data_test_path'])

    if process_train:
        print('Processing train dataset')
        dataset_train = JsonDataset(path=kwargs['data_train_path'], model=model, preload=kwargs['preload_data'])
        dataset_train.model = None
        pickle_it(dataset_train, os.path.join(kwargs['pickle_directory'], kwargs['train_pickle_name']))

    if process_validate:
        print('Processing validate dataset')
        dataset_val = JsonDataset(path=kwargs['data_validate_path'], model=model, preload=kwargs['preload_data'])
        dataset_val.model = None
        pickle_it(dataset_val, os.path.join(kwargs['pickle_directory'], kwargs['val_pickle_name']))

        dataset_bleu_val = BLEUDataset(path=kwargs['data_validate_path'], model=model)
        pickle_it(dataset_bleu_val, os.path.join(kwargs['pickle_directory'], kwargs['bleu_val_pickle_name']))

    if process_test:
        print('Processing test dataset')
        dataset_test = JsonDataset(path=kwargs['data_test_path'], model=model, preload=kwargs['preload_data'])
        dataset_test.model = None
        pickle_it(dataset_test, os.path.join(kwargs['pickle_directory'], kwargs['test_pickle_name']))

        dataset_bleu_test = BLEUDataset(path=kwargs['data_test_path'], model=model)
        pickle_it(dataset_bleu_test, os.path.join(kwargs['pickle_directory'], kwargs['bleu_test_pickle_name']))


if __name__ == '__main__':
    # preprocess train, test, val and pickle them
    param_path = './config.json'
    with open(param_path, 'rb') as f:
        parameters = json.load(f)
    main(**parameters)
