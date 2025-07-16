from preproc_datasplit import split
from preproc_ksl_vocab import main as generate_vocab
from preproc_pickle import main as process_pickle
from preproc_realign_annotations import realign
import json
import os


if __name__ == '__main__':
    param_path = './config.json'
    with open(param_path, 'rb') as f:
        params = json.load(f)

    print(f"Splitting data according to {params['data_split_path']}")
    split(**params)

    print(f"Re-aligning annotations")
    realign(['data/train/', 'data/validate/', 'data/test/'])

    print(f"\n\nGenerating KSL vocabulary from training file and saving to {params['ksl_vocab_path']}")
    generate_vocab(params['data_train_path'], ksl_vocab_path=params['ksl_vocab_path'])

    path = params['pickle_directory']
    train_f = params['train_pickle_name']
    val_f = params['val_pickle_name']
    bleu_val_f = params['bleu_val_pickle_name']
    test_f = params['test_pickle_name']
    bleu_test_f = params['bleu_test_pickle_name']
    print("\n\nPreprocessing data and pickling results to: {}, {}, {}, {}, and {}".format(
        os.path.join(path, train_f),
        os.path.join(path, val_f),
        os.path.join(path, test_f),
        os.path.join(path, bleu_val_f),
        os.path.join(path, bleu_test_f),
    ))
    process_pickle(**params)  # make sure this path matches dataset pickles above
