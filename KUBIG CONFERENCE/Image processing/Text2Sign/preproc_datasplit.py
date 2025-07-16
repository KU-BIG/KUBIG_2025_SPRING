import json
import os
import glob
from shutil import copyfile
from itertools import chain


def contains_from(s, xs):
    for x in xs:
        if x in s:
            return True
    return False


def copy_instance(src, dst, remove_kps=True):
    if remove_kps:
        try:
            with open(src, 'rb') as f:
                d = json.load(f)
        except Exception as e:
            print(f'Failed to load {src} with the following exception: \n{e}')
            return None
        if 'landmarks' in d.keys():
            d['landmarks'] = dict()
        with open(dst, 'w') as f:
            json.dump(d, f)
    else:
        copyfile(src, dst)


def prepare_directory(dir):
    """Creates directory if does not exist, removes all .json files from directory if exists"""
    if os.path.isdir(dir):
        print(f'Deleting .json files in {dir}...')
        files = glob.glob(f'{dir}*.json')
        for f in files:
            os.remove(f)
        print('Finished')
    else:
        print(f'Creating {dir}')
        os.mkdir(dir)


def split(remove_kps=True, **kwargs):
    groups_path = kwargs['group_data_path']

    with open(kwargs['data_split_path'], 'rb') as f:
        split_guide = json.load(f)

    data = list(chain.from_iterable([
        [os.path.join(p, f) for f in fs]
        for p, _, fs in os.walk(groups_path)
        if len(fs) > 0
    ]))

    copy_to_train = [file for file in data if contains_from(file, split_guide['train'])]
    copy_to_val = [file for file in data if contains_from(file, split_guide['val'])]
    copy_to_test = [file for file in data if contains_from(file, split_guide['test'])]

    print('Starting Train Data')
    prepare_directory(kwargs['data_train_path'])
    print('Copying files...')
    for file_i, file in enumerate(copy_to_train):
        print(f'{file_i}/{len(copy_to_train)}', end='\r')
        dst = os.path.join(kwargs['data_train_path'], os.path.split(file)[-1])
        copy_instance(file, dst, remove_kps=remove_kps)
    print('Finished Train Data')

    print('Starting Validation')
    prepare_directory(kwargs['data_validate_path'])
    print('Copying files...')
    for file_i, file in enumerate(copy_to_val):
        print(f'{file_i}/{len(copy_to_val)}', end='\r')
        dst = os.path.join(kwargs['data_validate_path'], os.path.split(file)[-1])
        copy_instance(file, dst, remove_kps=remove_kps)
    print('Finished Validation Data')

    print('Starting Test Data')
    prepare_directory(kwargs['data_test_path'])
    print('Copying files...')
    for file_i, file in enumerate(copy_to_test):
        print(f'{file_i}/{len(copy_to_test)}', end='\r')
        dst = os.path.join(kwargs['data_test_path'], os.path.split(file)[-1])
        copy_instance(file, dst, remove_kps=remove_kps)
    print('Finished Test Data')


if __name__ == '__main__':
    params_path = './config.json'
    with open(params_path, 'rb') as f:
        parameters = json.load(f)
    split(**parameters)
