import time
import torch
from typing import List, Union, Optional, Dict
import os
import json
import pdb
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
import random


def dict_to_device(data, device):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = dict_to_device(value, device)
    elif isinstance(data, torch.Tensor):
        data = data.to(device)
    elif isinstance(data, list):
        for i, x in enumerate(data):
            data[i] = dict_to_device(x, device)
    return data


def print_dict(
        d: Union[Dict, List],
        key: Optional[str] = None,
        spacing: Optional[str] = '',
        spacing_inc: Optional[str] = '    '
):
    """
    Pretty printing for dicts and lists.

    :param d: Dictionary or list to print
    :param key: Prints only the value accessed with key from d, defaults to None
    :param spacing: Used internally when called recursively, defaults to ''
    :param spacing_inc: The indent spacing for inner values, defaults to '    '
    """
    def space_print(spacing, val, key):
        if key is not None:
            print(val)
        else:
            print(spacing, val)

    if key is not None:
        print(spacing, '"{}"'.format(key), end=': ')
    if type(d) == dict:
        space_print(spacing=spacing, val='{', key=key)
        for k in d.keys():
            print_dict(d[k], key=k, spacing=spacing + spacing_inc)
        space_print(spacing=spacing, val='},', key=None)
    elif type(d) == list:
        if len(d) != 0:
            space_print(spacing=spacing, val='[', key=key)
            for ds in d:
                print_dict(ds, spacing=spacing + spacing_inc)
            space_print(spacing=spacing, val='],', key=None)
        else:
            space_print(spacing=spacing, val='[],', key=key)
    else:
        space_print(spacing=spacing, val=d, key=key)


def translate_test(
        model,
        test_sentences,
        max_length=None,
        device='cpu',
        return_translation=False,
        return_gloss_list=False,
        print_translation=True,
        print_gloss_list=True,
        print_time=True,
):
    model.eval()

    time_0 = time.time()
    with torch.no_grad():
        translation = model.batch_translate(test_sentences, max_length, device=device)['list']
    total_time = time.time() - time_0

    if return_gloss_list or print_gloss_list:
        gloss_list = [[item['gloss'] for item in trans if item['gloss'] != 'STEP'] for trans in translation]

    if print_translation:
        print_dict(translation)

    if print_gloss_list:
        print_dict(gloss_list)

    if print_time:
        print(f'Total translation time: {total_time}')

    out = tuple()
    if return_translation:
        out = (*out, translation)
    if return_gloss_list:
        out = (*out, gloss_list)
    return out


def sign_metric_fnc(pred, true):
    """
    Categorical Accuracy
    """
    pred = pred.argmax(dim=-1)
    here = torch.where(pred.eq(true))[0]
    return torch.tensor(here.shape).prod() / torch.tensor(pred.shape).prod()


def timing_metric_fnc(pred, true):
    """
    MSE
    """
    return (pred - true).pow(2).sum(dim=1).mean()


def load_sample_sentences(path, n=1):
    sample_files = [os.path.join(path, f) for f in os.listdir(path) if '.json' in f]
    assert len(sample_files) > 0, f"Unable to find any data samples in directory {path}."
    n = min(n, len(sample_files))
    test_sentences = []
    for i in range(n):
        with open(sample_files[i], 'rb') as f:
            test_sentences.append(json.load(f)['korean_text'])
    return test_sentences


def breakpoint():
    """For setting faster breakpoints"""
    pdb.set_trace()


@torch.no_grad()
def bleu_test(
        model,
        data,
        device,
        n_beams=1,
        top_k=None,
        top_p=None,
        max_length=512,
        channel_wise_bleu=False,
        ngram=4,
        weights=None,
        verbose=True,
):
    """Uses BLEU dataset only"""
    def extract_special(gloss):
        specials = [
            'x',
            '시',
            '시간',
            '날짜',
            'f',
        ]
        for special in specials:
            if special in gloss.split(':')[0]:
                return special
        return gloss

    def calculate(truths, pred, channel_wise_bleu):
        # replace fs with token representation and remove NMS for BLEU score calculation
        ignore = ['Ci', 'Hs', 'EBf', 'Hno', 'Mmo', 'Mo1', 'Tbt', 'Mctr']
        if channel_wise_bleu:
            res = 0
            for key in ['both', 'strong', 'weak']:
                res += corpus_bleu(
                    truths[key],
                    pred[key],
                    weights=weights,
                )
                # np.mean([
                #     sentence_bleu(truths[key][i], pred[key][i])
                #     if len(pred[key][i]) > 0 or any([len(ref) > 0 for ref in truths[key][i]])
                #     else 1
                #     for i in range(len(pred[key]))
                # ])
        else:
            truths = [[[extract_special(p) for p in row_ if p not in ignore] for row_ in row] for row in truths]
            pred = [[extract_special(p) for p in row if p not in ignore] for row in pred]

            return corpus_bleu(
                truths,
                pred,
                weights=weights,
            )
    if weights is None:
        weights = [1/ngram for _ in range(ngram)]
    model.eval()
    truths = [] if not channel_wise_bleu else {'both': [], 'strong': [], 'weak': []}
    pred = [] if not channel_wise_bleu else {'both': [], 'strong': [], 'weak': []}
    print('\nGenerating translations...')
    nmbs = len(data)
    score = 0
    for mb, d in enumerate(data):
        print('Evaluating mb {}/{}  {}'.format(mb + 1, nmbs, score if verbose else ''), end='\r')
        d = dict_to_device(d, device)

        if channel_wise_bleu:
            for key in ['both', 'strong', 'weak']:
                truths[key].extend([
                    [
                        [g.split('#')[0] for g in gloss_list]
                        for gloss_list in row[key]
                    ] for row in d['glosses_channel']
                ])
        else:
            truths.extend([
                [
                    [g.split('#')[0] for g in gloss_list]
                    for gloss_list in row
                ] for row in d['glosses']
            ])

        translation = model.batch_translate(
            sequences=d['source_text'],
            max_length=max_length,
            n_beams=n_beams,
            top_k=top_k,
            top_p=top_p,
            device=device,
        )['list']

        if channel_wise_bleu:
            pred['both'].extend([
                [item['gloss'] for item in trans if (item['gloss'] not in ['STEP', 'UNK'] and item['mode'] == 'BOTH')]
                for trans in translation
            ])
            pred['strong'].extend([
                [item['gloss'] for item in trans if (item['gloss'] not in ['STEP', 'UNK'] and item['mode'] == 'STRONG')]
                for trans in translation
            ])
            pred['weak'].extend([
                [item['gloss'] for item in trans if (item['gloss'] not in ['STEP', 'UNK'] and item['mode'] == 'WEAK')]
                for trans in translation
            ])
        else:
            pred_glosses = [
                [item['gloss'] for item in trans if item['gloss'] not in ['STEP', 'UNK']]
                for trans in translation
            ]
            pred.extend(pred_glosses)
        if verbose:
            score = calculate(truths, pred, channel_wise_bleu)

    score = calculate(truths, pred, channel_wise_bleu)

    return score


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class UnfreezeScheduler:
    def __init__(self, optimizer, plateau_scheduler, param_norm, *, param_steps):
        self.optimizer = optimizer
        self.scheduler = plateau_scheduler
        self.param_norm = param_norm
        self.finished = dict()
        self.params = dict()
        for param_group, step in param_steps:
            if step in self.params.keys():
                self.params[step].append(param_group)
            else:
                self.params[step] = [param_group]
        self.counter = 0
        self.has_params_to_add = len(self.params) > 0

    def step(self, steps=1):
        if self.has_params_to_add:
            self.counter += steps
            keys = [key for key in self.params.keys() if key <= self.counter]
            for key in keys:
                for param_group in self.params[key]:
                    self.optimizer.add_param_group(param_group)
                    self.scheduler.min_lrs.append(min(self.scheduler.min_lrs))
                    self.param_norm.add_params(param_group)
                self.finished[key] = self.params.pop(key)
            if len(self.params) == 0:
                self.has_params_to_add = False


def init_optimizer(optimizer, params, lr):
    if optimizer == 'sgd':
        optim = torch.optim.SGD(
            params,
            lr=lr,
            momentum=0.9,
        )
    elif optimizer == 'adam':
        optim = torch.optim.Adam(
            params,
            lr=lr,
        )
    elif optimizer == 'adamw':
        optim = torch.optim.AdamW(
            params,
            lr=lr,
        )
    else:
        raise ValueError
    return optim
