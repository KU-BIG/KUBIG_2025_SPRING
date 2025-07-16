import os
from random import shuffle
import json
from itertools import chain
import re
from collections import defaultdict


def get_all_files(path):
    files = [
        [file for file in files if '.json' in file]
        for root, dirs, files in os.walk(path) if files != []
    ]
    return list(chain.from_iterable(files))


def group_by_category(files):
    grouped = defaultdict(list)
    for file in files:
        category = re.split(u'(?<=[a-zA-Z])\d', file.split('_')[3])[0]
        grouped[category].append(file)
    return list(grouped.values())


def remain_remove_bybase(ps, bs):
    remove = []
    remain = []
    for p in ps:
        if os.path.basename(p).replace('.json', '') in bs:
            remove.append(p)
        else:
            remain.append(p)
    return remain, remove


def group_by_base(files, group_base_flag=True):
    bases_map = defaultdict(list)
    if group_base_flag:
        for file in files:
            bases_map[file.split('_')[3]].append(file)
        bases, bases_map = list(bases_map.keys()), bases_map
    else:
        bases, bases_map = files, {file: [file] for file in files}

    return bases, bases_map


def split(files, ratio=(.8, .1, .1), put_train=None, group_base_flag=True):
    """
    group_base_flag == True => 1:2 and 1:3 files with same source sentence are put into the same data subset.
    """
    train = []
    validate = []
    test = []
    for group in files:
        if put_train is not None:
            group, add_train = remain_remove_bybase(group, put_train)
            train.extend(add_train)

        bases, bases_map = group_by_base(group, group_base_flag=group_base_flag)
        if len(bases) >= 3:
            shuffle(bases)
            train_i = max(1, round(len(bases) * ratio[0]))
            validate_i = max(1, round(len(bases) * ratio[1]))
            test_i = max(1, round(len(bases) * ratio[2]))
            delta = len(bases) - (train_i + test_i + validate_i)
            if delta < 0:
                train_i -= delta  # assume that ratio[0] > ratio[1 or 2] percentage is greatest
            elif delta > 0:
                train_i += delta

            train.extend(list(chain.from_iterable(
                [bases_map[base] for base in bases[:train_i]]
            )))
            validate.extend(list(chain.from_iterable(
                [bases_map[base] for base in bases[train_i: train_i + validate_i]]
            )))
            test.extend(list(chain.from_iterable(
                [bases_map[base] for base in bases[train_i + validate_i:]]
            )))
        else:
            train.extend(list(chain.from_iterable(list(bases_map.values()))))

    return train, validate, test


if __name__ == '__main__':
    with open('./config.json', 'rb') as f:
        parameters = json.load(f)

    data_split_path = parameters['data_split_path']

    source_path = parameters['group_data_path']

    files_all = get_all_files(source_path)

    files_bycategory = group_by_category(files_all)

    train, validate, test = split(files_bycategory, group_base_flag=parameters['split_group_by_base'])

    total = len(train) + len(validate) + len(test)
    max_str_len = max(len(str(len(train))), len(str(len(validate))), len(str(len(test))))
    output = '\nData subset sizes...\nTrain:    {} {}({}%)\nValidate: {} {}({}%)\nTest:     {} {}({}%)\n------------\nTotal:    {}'.format(
        len(train),
        ' ' * (max_str_len - len(str(len(train)))),
        str(len(train) / total * 100)[:6],
        len(validate),
        ' ' * (max_str_len - len(str(len(validate)))),
        str(len(validate) / total * 100)[:6],
        len(test),
        ' ' * (max_str_len - len(str(len(test)))),
        str(len(test) / total * 100)[:6],
        total,
    )
    print(output)

    with open(data_split_path, 'w') as f:
        json.dump({'train': train, 'val': validate, 'test': test}, f)
