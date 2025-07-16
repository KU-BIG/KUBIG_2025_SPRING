import torch
import json
from utils import dict_to_device
import os
import math
import string
import pandas as pd


def dict_get(d, k, default=None):
    if k in d.keys():
        return d[k]
    return default


def find_parsing(xs, key=''):
    temp = [x_i for x_i, x in enumerate(xs) if x['BlockValue'] == '-1' and len(x['BlockValueText'].split(':')) > 0]
    temp = [
        (
            i,
            xs[i]['BlockValueText'].split(':')[1],
            xs[i]['StartTime'],
            xs[i]['EndTime'],
        )
        for i in temp
        if xs[i]['BlockValueText'].split(':')[0] == key
    ]
    return temp


def extract_time(s):
    if '시' in s and '분' in s:
        hour, minute = s.replace('시', '-').replace('분', '-').split('-')[:2]
    elif '분' in s:
        hour = ''
        minute = s.replace('분', '')
    elif '시' in s:
        hour = s.replace('시', '')
        minute = ''
    else:  # failed parsing
        hour = s
        minute = ''

    return hour, minute


def extract_date(s):
    if '월' in s and '일' in s:
        month, day = s.replace('월', '-').replace('일', '-').split('-')[:2]
    elif '일' in s:
        month = ''
        day = s.replace('일', '')
    elif '월' in s:
        month = s.replace('월', '')
        day = ''
    else:  # failed parsing
        month = s
        day = ''

    return month, day


def strip_punct(s):
    return s.translate(str.maketrans('', '', string.punctuation))


def post_translate_parse(d):
    """
    For splitting dates, times, etc before sending to the player. The demo player for this project does not fully
    support all animations, especially for numbers. Ex: minutes are treated like a string and are fingerspelled.
    """
    month_map = {
        '1': '4792',
        '2': '13530',
        '3': '13301',
        '4': '26128',
        '5': '13319',
        '6': '13937',
        '7': '11998',
        '8': '13280',
        '9': '13358',
        '10': '24341',
        '11': '13333',
        '12': '24136',
    }
    hour_map = {
        '1': '24467',
        '2': '24468',
        '3': '24469',
        '4': '24470',
        '5': '24471',
        '6': '24472',
        '7': '24473',
        '8': '24474',
        '9': '24475',
        '10': '24476',
        '11': '24477',
        '12': '24478',
        '13': '24467',
        '14': '24468',
        '15': '24469',
        '16': '24470',
        '17': '24471',
        '18': '24472',
        '19': '24473',
        '20': '24474',
        '21': '24475',
        '22': '24476',
        '23': '24477',
        '24': '24478',
    }
    keys = ['SequenceHandBoth', 'SequenceHandRight', 'SequenceHandLeft']

    # numbers (convert to fingerspelling)
    for key in keys:
        for item in d[key]:
            if item['BlockValue'] == '-1' and item['BlockType'] == 'animation':
                try:
                    if str(int(strip_punct(item['BlockValueText']))) == strip_punct(item['BlockValueText']):
                        item['BlockValue'] = item['BlockValueText']
                        item['BlockType'] = 'fingerspell'
                except:
                    continue

    # months
    months = [(key, find_parsing(d[key], '날짜')) for key in keys]
    months = [item for item in months if len(item[1]) > 0]
    months = [(item[0], sorted(item[1], key=lambda x: x[2])) for item in months]
    for key, arrs in months[::-1]:
        if key not in ['SequenceHandBoth']:  # for now, skip if not in both
            continue
        for mon in arrs[::-1]:
            month, day = extract_date(d[key][mon[0]]['BlockValueText'].split(':')[1])
            month_id = dict_get(month_map, month, default='-1')
            time_split = (mon[3] + mon[2]) / 2
            new_item = {
                'StartTime': time_split,
                'EndTime': mon[3],
                'BlockType': 'fingerspell',
                'BlockValue': str(day),
                'BlockValueText': str(day),
                'BlockExtraData': [],
            }
            d[key][mon[0]]['BlockValue'] = str(month_id)
            d[key][mon[0]]['BlockValueText'] = str(month)
            if day != '':
                right_pos = [item_i for item_i, item in enumerate(d['SequenceHandRight']) if item['StartTime'] > new_item['EndTime']]
                if len(right_pos) > 0:
                    d['SequenceHandRight'].insert(right_pos[0], new_item)
                else:
                    d['SequenceHandRight'].append(new_item)


    # times
    times = [(key, find_parsing(d[key], '시')) for key in keys]
    times = [item for item in times if len(item[1]) > 0]
    times = [(item[0], sorted(item[1], key=lambda x: x[2])) for item in times]
    for key, arrs in times:
        for tm in arrs[::-1]:
            hour, minute = extract_time(d[key][tm[0]]['BlockValueText'].split(':')[1])
            hour_id = dict_get(hour_map, hour, default='-1')
            time_split = (tm[3] + tm[2]) / 2
            new_item = {
                'StartTime': time_split,
                'EndTime': tm[3],
                'BlockType': 'fingerspell',
                'BlockValue': str(minute),
                'BlockValueText': str(minute),
                'BlockExtraData': [],
            }
            d[key][tm[0]]['EndTime'] = time_split
            d[key][tm[0]]['BlockValue'] = str(hour_id)
            d[key][tm[0]]['BlockValueText'] = str(hour)

            if minute != '':  # add to end of right hand sequence
                d['SequenceHandRight'].append(new_item)
                # if key == 'SequenceHandStrong':
                #     d[key].insert(tm[0] + 1, new_item)
                # else:
                #     right_pos = [item_i for item_i, item in enumerate(d['SequenceHandRight']) if item['StartTime'] > new_item['EndTime']]
                #     if len(right_pos) > 0:
                #         d['SequenceHandRight'].insert(right_pos[0], new_item)
                #     else:
                #         d['SequenceHandRight'].append(new_item)

    # reorder
    for key in keys:
        d[key] = sorted(d[key], key=lambda x: x['StartTime'])
    # durations

    # make sure nothing is touching
    for key in ['SequenceHandBoth', 'SequenceHandRight', 'SequenceHandLeft', 'SequenceBody', 'SequenceFace', 'SequenceMouth']:
        n = len(d[key]) - 1
        for i in range(n):
            if d[key][i + 1]['StartTime'] - d[key][i]['EndTime'] <= .01:
                d[key][i + 1]['StartTime'] += min((d[key][i + 1]['EndTime'] - d[key][i + 1]['StartTime']) / 2, .1)
    return d


@torch.no_grad()
def translate(model, data, device, force_ids=False):
    model.eval()
    translations_player = []
    translations_nia = []
    print('\nGenerating translations...')
    nmbs = len(data)
    for mb, d in enumerate(data):
        print('Evaluating mb {}/{}'.format(mb + 1, nmbs), end='\r')
        d = dict_to_device(d, device)
        trans = model.batch_translate(
            sequences=d['source_text'],
            max_length=512,
            device=device,
            nia_data_script=True,
            player_script=True,
        )
        if force_ids:
            for res_i, res in enumerate(trans['nia_data_script']):
                if 'metadata' not in res.keys():
                    res['metadata'] = {}
                res['metadata']['id'] = d['nia_id'][res_i]
            for res_i, res in enumerate(trans['player_script']):
                res['NiaID'] = d['nia_id'][res_i]
                post_translate_parse(res)

        translations_player.extend(trans['player_script'])
        translations_nia.extend(trans['nia_data_script'])
    return {'player_script': translations_player, 'nia_data_script': translations_nia}


def save_translations(translations, path, save_summary=False, save_as_nia_id=True):
    if not os.path.isdir(path):
        os.mkdir(path)
    if len(translations) > 9:
        figs = math.ceil(math.log(len(translations), 10))
    else:
        figs = 1
    base = '0' * figs
    print('saving               ')
    data = []
    for t_i, translation in enumerate(translations):
        print(f'Saving {t_i}/{len(translations) - 1}', end='\r')
        if save_as_nia_id and save_summary:
            idx = f"{translation['NiaID']}_EQ"
        else:
            idx = base
            idx = idx[:-len(str(t_i))] + str(t_i)
            # idx = f'{idx}.json'
        if save_summary:
            label = f"{translation['NiaID']}.json"

        with open(os.path.join(path, f'{idx}.json'), 'w') as f:
            json.dump(translation, f)

        if save_summary:
            data.append({
                '#ID': t_i,
                '한글 문장': translation['SourceText'].replace('"', "'"),
                '번역결과(json)': f'{idx}.json',
                '영상 파일 이름': f'{idx}.mp4',
                '로그 파일 이름(json 형식)': f'{idx}_log.txt',
                'NIA ID': translation['NiaID'],
                '정답(json)': label,
                'Bleu Score': 0,
                'Video': 'true',
            })
    if save_summary:
        pd.DataFrame.from_dict(data).to_csv(os.path.join(path, 'NIASetting.csv'), index=False)
