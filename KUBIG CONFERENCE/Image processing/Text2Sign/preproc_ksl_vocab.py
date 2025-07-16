import json
import os
import sys


# call python generate_ksl_vocab.py [PATH TO TRAINING DATASET]


def main(*paths, ksl_vocab_path='./ksl_vocab.json'):
    sign_keys = [
        'sign_gestures_both',
        'sign_gestures_strong',
        'sign_gestures_weak',
    ]
    nms_vocab = [
        "Mmo",
        "Hno",
        "Mo1",
        "Hs",
        "EBf",
        "Mctr",
        "Ci",
        "Tbt",
    ]

    files = []
    for path in paths:
        files_ = os.listdir(path)
        files.extend([f'{path}/{f}' for f in files_ if '.json' in f])

    glosses = []
    for file_i, file in enumerate(files):
        print(f'Processing file: {file_i}', end='\r')
        with open(file, 'rb') as f:
            d = json.load(f)
        try:
            d = d['sign_script']
            for key in sign_keys:
                for item in d[key]:
                    try:
                        # make vocab entry for s
                        if item['express'] == 's':
                            # glosses.append(item['gloss_id'])
                            gloss = item['gloss_id'].split('#')[0]
                            if '0' in gloss or gloss == '':  # ignore incorrectly formatting glosses
                                continue
                            glosses.append(gloss)

                        # make vocab entry for d (for now, just add one for each -- no special parsing)
                        if item['express'] == 'd':
                            gloss = item['gloss_id']
                            glosses.append(gloss)

                        # other types (n, s) can  use pointing (no need to generate token)
                    except Exception as e:
                        print(e)
        except Exception as e:
            print(f'Failed to process file {file} with error: \n{e}')
    glosses = list(set(glosses))
    vocab = {
        'sign_vocab': glosses,
        'nms_vocab': nms_vocab,
    }

    print(f'Saving new vocab file: {ksl_vocab_path}')
    with open(ksl_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("""\nCould not find any specified paths. Call this function using the following to specify a training directory:
    python generate_ksl_vocab.py [PATH TO DATASET] [PATH TO DATASET] ...
    Defaulting to 'data/train/'""")
        paths = ['data/train/']
    else:
        paths = sys.argv[1:]

    for path in paths:
        assert os.path.isdir(path), "Could not find the specified data directory. Please make check the path."

    with open('./config.json', 'rb') as f:
        ksl_vocab_path = json.load(f)['ksl_vocab_path']

    main(*paths, ksl_vocab_path=ksl_vocab_path)
