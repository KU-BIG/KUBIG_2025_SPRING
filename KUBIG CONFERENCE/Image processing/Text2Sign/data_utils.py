from torch.utils.data import Dataset
import os
import json
from typing import Optional
import torch
import torch.nn as nn
from itertools import chain


class JsonDataset(Dataset):
    """
    Dataset for loading json data with keys "source", "suji", and "bisuji".

    :param path: Path to folder containing json files
    :type path: str
    :param preload: True: data is pre-loaded into memory, False: data is read online each batch, defaults to True
    :type preload: bool, optional
    """
    def __init__(
            self,
            path: str,
            model: nn.Module,
            preload: Optional[bool] = True,
    ):
        super(JsonDataset, self).__init__()
        self.files = os.listdir(path)
        self.files = ['{}/{}'.format(path, f) for f in self.files if '.json' in f]
        self.preload = preload
        self.model = model
        self.no_parse = []
        if self.preload:
            data = []
            for file_i, file in enumerate(self.files):
                print(f'Processing file {file_i + 1}/{len(self.files)}', end='\r')
                try:  # skip if parsing fails
                    with open(file, 'rb') as f:
                        data.append(self.model.process_instance(json.load(f)))
                except:
                    self.no_parse.append(file)
            self.files = data
        print(f'Skipped {len(self.no_parse)} files due to parsing errors: \n{self.no_parse}\n\n')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        """
        if self.preload:
            return self.files[idx]
        else:
            with open(self.files[idx], 'rb') as f:
                file = json.load(f)
            return self.model.process_instance(file)


class CollateWrapper:
    """
    Class to allow calling collate function with reference to a model with data formatting method.

    :param model: Model with method named make_training_array that will format data into the appropriate shape
    :type model: class: torch.nn.Module
    """
    def __init__(self, model: torch.nn.Module):
        self.model = model

    # def collate_old(self, batch):
    #     source = [b['text'] for b in batch]
    #     start_here = [len(text) for text in source]
    #     source_len = max(start_here)
    #     source_len = min(source_len, self.model.max_source_length)
    #     target_len = max([len(b['types']) for b in batch])
    #     # target_len = min(target_len, self.model.max_target_length)  # should already be < max len + 1
    #
    #     # pad source
    #     source = [text[:source_len] + [0] * max(0, source_len - len(text)) for text in source]
    #     source = torch.tensor(source)
    #     source_inds = list(zip(*[([i] * s, torch.arange(s)) for i, s in enumerate(start_here)]))
    #     source_inds[0] = list(chain(source_inds[0]))
    #     source_inds[1] = torch.cat(source_inds[1])
    #
    #     # flatten targets (use flat and fill in using indices below)
    #     types = torch.tensor(list(chain([b['types'] for b in batch])))
    #     signs = torch.tensor(list(chain([b['signs'] for b in batch])))
    #     nms = torch.tensor(list(chain([b['nmss'] for b in batch])))
    #     token_durations = torch.tensor(list(chain([b['token_durations'] for b in batch])))
    #     step_durations = torch.tensor(list(chain([b['step_durations'] for b in batch])))
    #     # types = [b['types'][:target_len] + [0] * target_len - len(b['types']) for i, b in enumerate(batch)]
    #     # signs = [b['signs'] for i, b in enumerate(batch)]
    #     # nms = [b['nms'] for i, b in enumerate(batch)]
    #     # token_durations = [b['token_durations'] for b in batch]
    #     # step_durations = [b['step_durations'] for b in batch]
    #
    #     # indices for slicing prediction (for loss calculation)
    #     type_inds = list(zip(*[([i] * len(b['types']), torch.arange(len(b['types']))) for i, b in enumerate(batch)]))
    #     type_inds[0] = list(chain(type_inds[0]))
    #     target_offset = torch.tensor(list(chain([[start_here[i]] * len(inds) for i,inds in enumerate(type_inds[1])])))
    #     type_inds[1] = torch.cat(type_inds[1])
    #     sign_inds = list(zip(*[([i] * len(b['sign_inds']), torch.tensor(b['sign_inds'])) for i, b in enumerate(batch)]))
    #     sign_inds[0] = list(chain(sign_inds[0]))
    #     sign_inds[1] = torch.cat(sign_inds[1])
    #     nms_inds = list(zip(*[([i] * len(b['nms_inds']), torch.tensor(b['nms_inds'])) for i, b in enumerate(batch)]))
    #     nms_inds[0] = list(chain(nms_inds[0]))
    #     nms_inds[1] = torch.cat(nms_inds[1])
    #     ksl_inds = tuple()
    #     ksl_inds[0] = sign_inds[0] + nms_inds[0]
    #     ksl_inds[1] = torch.cat([sign_inds[1], nms_inds[1]])
    #     token_inds = list(zip(*[([i] * len(b['token_inds']), torch.tensor(b['token_inds'])) for i, b in enumerate(batch)]))
    #     token_inds[0] = list(chain(token_inds[0]))
    #     token_inds[1] = torch.cat(token_inds[1])
    #     step_inds = list(zip(*[([i] * len(b['step_inds']), torch.tensor(b['step_inds'])) for i, b in enumerate(batch)]))
    #     step_inds[0] = list(chain(step_inds[0]))
    #     step_inds[1] = torch.cat(step_inds[1])
    #
    #     # build padding mask
    #     padding_mask = []
    #
    #     data = {
    #         'source': source,
    #         'source_inds': source_inds,
    #         'target_offset': target_offset,
    #         'max_source': source_len,
    #         'max_target': target_len,
    #         'types': types,
    #         'signs': signs,
    #         'nms': nms,
    #         'token_durations': token_durations,
    #         'step_durations': step_durations,
    #         'type_inds': type_inds,
    #         'ksl_inds': ksl_inds,
    #         'sign_inds': sign_inds,
    #         'nms_inds': nms_inds,
    #         'token_inds': token_inds,
    #         'step_inds': step_inds,
    #         'padding_mask': padding_mask,
    #     }
    #
    #     return data

    def collate(self, batch):
        source_text = [b['source_text'] for b in batch]
        gloss_list = [b['gloss_list'] for b in batch]

        source = [b['source_tokens'] for b in batch]
        start_here = [len(text) for text in source]
        source_len = max(start_here)
        source_len = min(source_len, self.model.max_source_length)
        target_len = max([len(b['types']) for b in batch])
        # target_len = min(target_len, self.model.max_target_length)  # should already be < max len + 1

        # pad source
        source = [text[:source_len] + [0] * max(0, source_len - len(text)) for text in source]
        source = torch.tensor(source)
        source_inds = list(zip(*[([i] * s, torch.arange(s)) for i, s in enumerate(start_here)]))
        source_inds[0] = list(chain.from_iterable(source_inds[0]))
        source_inds[1] = torch.cat(source_inds[1])

        # flatten targets (use flat and fill in using indices below)
        types = torch.tensor(list(chain.from_iterable([b['types'] for b in batch])))
        types_nobos = torch.tensor(list(chain.from_iterable([b['types'][1:] for b in batch])))
        tokens = torch.tensor(list(chain.from_iterable([b['tokens'] for b in batch])))
        durations = torch.tensor(list(chain.from_iterable([b['durations'] for b in batch])))
        durations_nobos = torch.tensor(list(chain.from_iterable([b['durations'][1:] for b in batch])))
        signs = torch.tensor(list(chain.from_iterable([b['signs'] for b in batch])))
        nms = torch.tensor(list(chain.from_iterable([b['nmss'] for b in batch])))
        ksl = torch.cat([signs, nms]).long()
        token_durations = torch.tensor(list(chain.from_iterable([b['token_durations'] for b in batch])))
        step_durations = torch.tensor(list(chain.from_iterable([b['step_durations'] for b in batch])))
        # print([b['point'] for b in batch])
        # point = torch.tensor(list(chain.from_iterable(b['point'] for b in batch)))
        # point = torch.tensor(list(chain.from_iterable([list(p) for b in batch for p in b['point']])))
        ps = [list(p) for b in batch for p in b['point']]
        point = torch.zeros(len(ps), source_len).float()
        for p_i, p in enumerate(ps):
            point[p_i][p] = 1
        # types = [b['types'][:target_len] + [0] * target_len - len(b['types']) for i, b in enumerate(batch)]
        # signs = [b['signs'] for i, b in enumerate(batch)]
        # nms = [b['nms'] for i, b in enumerate(batch)]
        # token_durations = [b['token_durations'] for b in batch]
        # step_durations = [b['step_durations'] for b in batch]

        # indices for slicing prediction (for loss calculation)
        type_inds = list(zip(*[([i] * len(b['types']), torch.arange(len(b['types']))) for i, b in enumerate(batch)]))
        type_inds[0] = list(chain.from_iterable(type_inds[0]))
        target_offset = torch.tensor(list(chain.from_iterable([[start_here[i]] * len(inds) for i, inds in enumerate(type_inds[1])])))
        type_inds[1] = torch.cat(type_inds[1])

        type_inds_noeos = list(zip(*[([i] * (len(b['types']) - 1), torch.arange(len(b['types']) - 1)) for i, b in enumerate(batch)]))
        type_inds_noeos[0] = list(chain.from_iterable(type_inds_noeos[0]))
        target_offset_noeos = torch.tensor(
            list(chain.from_iterable([[start_here[i]] * len(inds) for i, inds in enumerate(type_inds_noeos[1])]))
        ).long()
        type_inds_noeos[1] = torch.cat(type_inds_noeos[1]).long()

        type_inds_nobos = list(zip(*[([i] * (len(b['types']) - 1), torch.arange(1, len(b['types']))) for i, b in enumerate(batch)]))
        type_inds_nobos[0] = list(chain.from_iterable(type_inds_nobos[0]))
        target_offset_nobos = torch.tensor(
            list(chain.from_iterable([[start_here[i]] * len(inds) for i, inds in enumerate(type_inds_nobos[1])]))
        )
        type_inds_nobos[1] = torch.cat(type_inds_nobos[1])

        sign_inds = list(zip(*[([i] * len(b['sign_inds']), torch.tensor(b['sign_inds'])) for i, b in enumerate(batch)]))
        sign_inds[0] = list(chain.from_iterable(sign_inds[0]))
        nms_inds = list(zip(*[([i] * len(b['nms_inds']), torch.tensor(b['nms_inds'])) for i, b in enumerate(batch)]))
        nms_inds[0] = list(chain.from_iterable(nms_inds[0]))

        ksl_offset = torch.cat([
            torch.tensor(list(chain.from_iterable([[start_here[i]] * len(inds) for i, inds in enumerate(sign_inds[1])]))),
            torch.tensor(list(chain.from_iterable([[start_here[i]] * len(inds) for i, inds in enumerate(nms_inds[1])])))
        ]).long()
        sign_inds[1] = torch.cat(sign_inds[1])
        nms_inds[1] = torch.cat(nms_inds[1])
        ksl_inds = [sign_inds[0] + nms_inds[0], torch.cat([sign_inds[1], nms_inds[1]]).long()]

        token_inds = list(zip(*[([i] * len(b['token_inds']), torch.tensor(b['token_inds'])) for i, b in enumerate(batch)]))
        token_inds[0] = list(chain.from_iterable(token_inds[0]))
        token_inds[1] = torch.cat(token_inds[1])
        step_inds = list(zip(*[([i] * len(b['step_inds']), torch.tensor(b['step_inds'])) for i, b in enumerate(batch)]))
        step_inds[0] = list(chain.from_iterable(step_inds[0]))
        step_inds[1] = torch.cat(step_inds[1])
        point_inds = list(zip(*[([i] * len(b['point_inds']), torch.tensor(b['point_inds'])) for i, b in enumerate(batch)]))
        point_inds[0] = list(chain.from_iterable(point_inds[0]))
        point_offset = torch.tensor([start_here[i] for i in point_inds[0]])
        point_inds[1] = torch.cat(point_inds[1]) + point_offset

        # build padding mask
        # total_lengths = [start_here[row] + 1 + len(batch[row]['types']) + 1 for row in range(len(source))]
        total_lengths = [start_here[row] + len(batch[row]['types']) for row in range(len(source))]
        total_len_max = source_len + target_len  #  + 2
        padding_mask = torch.tensor([
            total_lengths[row] * [1] + (total_len_max - total_lengths[row]) * [0]
            for row in range(len(source))
        ])

        data = {
            'source_text': source_text,
            'gloss_list': gloss_list,
            'source': source,
            'source_inds': source_inds,
            'target_offset': target_offset,
            'max_source': source_len,
            'max_target': target_len,
            'types': types,
            'types_nobos': types_nobos,
            'tokens': tokens,
            'signs': signs,
            'nms': nms,
            'ksl': ksl,
            'ksl_offset': ksl_offset,
            'durations': durations,
            'durations_nobos': durations_nobos,
            'token_durations': token_durations,
            'step_durations': step_durations,
            'type_inds': type_inds,
            'ksl_inds': ksl_inds,
            'sign_inds': sign_inds,
            'nms_inds': nms_inds,
            'token_inds': token_inds,
            'step_inds': step_inds,
            'padding_mask': padding_mask,
            'type_inds_noeos': type_inds_noeos,
            'target_offset_noeos': target_offset_noeos,
            'type_inds_nobos': type_inds_nobos,
            'target_offset_nobos': target_offset_nobos,
            'point': point,
            'point_inds': point_inds,
        }

        return data


class JsonDatasetText(Dataset):
    """
    Dataset for loading json data with keys "source", "suji", and "bisuji".

    :param path: Path to folder containing json files
    :type path: str
    :param preload: True: data is pre-loaded into memory, False: data is read online each batch, defaults to True
    :type preload: bool, optional
    """
    def __init__(
            self,
            path: str,
            specific_paths: Optional[list] = None,
            preload: Optional[bool] = True,
    ):
        super(JsonDatasetText, self).__init__()

        if specific_paths is None:
            self.files = os.listdir(path)
            self.files = ['{}/{}'.format(path, f) for f in self.files if '.json' in f]
        else:
            self.files = specific_paths

        self.preload = preload
        no_parse = []
        if self.preload:
            data = []
            for file_i, file in enumerate(self.files):
                print(f'Parsing file: {file_i + 1}/{len(self.files)}', end='\r')
                try:  # skip if loading fails
                    with open(file, 'rb') as f:
                        d = json.load(f)
                    data.append({'source_text': d['korean_text'], 'nia_id': d['metadata']['id']})
                except:
                    no_parse.append(file)
            self.files = data
        print(f'Skipped {len(no_parse)} files due to parsing errors: \n{no_parse}\n\n')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        """
        if self.preload:
            return self.files[idx]
        else:
            with open(self.files[idx], 'rb') as f:
                d = json.load(f)
            return {'source_text': d['korean_text'], 'nia_id': d['metadata']['id']}

    def collate(self, batch):
        return {
            'source_text': [b['source_text'] for b in batch],
            'nia_id': [b['nia_id'] for b in batch],
        }


class BLEUDataset(Dataset):
    """
    Dataset for multi-target data (for calculating BLEU score)

    """
    def __init__(
            self,
            path: str,
            model: nn.Module,
            **kwargs,
    ):
        super(BLEUDataset, self).__init__()
        files = [f for f in os.listdir(path) if f.split('.')[-1] == 'json']
        from collections import defaultdict
        self.data = defaultdict(lambda: defaultdict(list))
        self.channel_data = defaultdict(lambda: defaultdict(list))
        self.no_parse = []
        for file in files:
            try:  # skip if parsing fails
                with open(os.path.join(path, file), 'rb') as f:
                    parsed = model.process_instance(json.load(f))
                assert isinstance(parsed['source_text'], str) and isinstance(parsed['gloss_list'], list)
                self.data[self.split_id(file)][parsed['source_text']].append(parsed['gloss_list'])

                self.channel_data[parsed['source_text']]['both'].append(parsed['script']['BOTH'])
                self.channel_data[parsed['source_text']]['strong'].append(parsed['script']['STRONG'])
                self.channel_data[parsed['source_text']]['weak'].append(parsed['script']['WEAK'])

            except:
                self.no_parse.append(file)
        print(f'Skipped {len(self.no_parse)} files due to parsing errors: \n{self.no_parse}\n\n')
        # self.keys = list(self.data.keys())
        self.data = [(list(ds.keys())[0], ds[list(ds.keys())[0]]) for ds in list(self.data.values())]

        self.channel_data = [(s, d) for s, d in list(self.channel_data.items())]

    def split_id(self, file_name):
        return file_name.split('_')[3]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        """
        return {
            'source_text': self.data[idx][0],
            'glosses': self.data[idx][1],
            'glosses_channel': self.channel_data[idx][1],
        }

    def collate(self, batch):
        source_text = [b['source_text'] for b in batch]
        glosses = [b['glosses'] for b in batch]
        glosses_channel = [b['glosses_channel'] for b in batch]
        return {
            'source_text': source_text,
            'glosses': glosses,
            'glosses_channel': glosses_channel
        }
