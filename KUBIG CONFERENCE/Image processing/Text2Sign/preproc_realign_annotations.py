#### find (and correct) misaligned signs
import json
import os


def realign_file(data, threshold=0.1):
    insert = []
    s_del = []
    w_del = []
    for s_i, s in enumerate(data['sign_script']['sign_gestures_strong']):
        for w_i, w in enumerate(data['sign_script']['sign_gestures_weak']):
            if s['gloss_id'] == w['gloss_id']:
                if abs(s['start'] - w['start']) < threshold and abs(s['end'] - w['end']) < threshold:
                    s_del.append(s_i)
                    w_del.append(w_i)
                    # cutoff start and end at the overlap (to be safe)
                    s['start'] = max(s['start'], w['start'])
                    s['end'] = min(s['end'], w['end'])
                    insert.append(s)
    if len(insert) == 0:
        return None, False

    data['sign_script']['sign_gestures_both'] += insert
    data['sign_script']['sign_gestures_both'] = sorted(
        data['sign_script']['sign_gestures_both'],
        key=lambda x: x['start'],
    )

    s_del = sorted(list(set(s_del)), reverse=True)
    for i in s_del:
        data['sign_script']['sign_gestures_strong'].pop(i)
    w_del = sorted(list(set(w_del)), reverse=True)
    for i in w_del:
        data['sign_script']['sign_gestures_weak'].pop(i)
    return data, True


def realign(paths):
    files = []
    for path in paths:
        files.extend([os.path.join(path, f) for f in os.listdir(path)])
    n = 0
    for file_i, file in enumerate(files):
        print(f'{file_i}/{len(files)}', end='\r')
        try:
            with open(file, 'rb') as f:
                d = json.load(f)
        except Exception as e:
            print(f'Cannot load {file}\nError: {e}')
            continue
        try:
            d, realigned = realign_file(d)
        except Exception as e:
            print(f'Cannot realign {file}\nError: {e}')
            continue
        if realigned:
            with open(file, 'w', encoding='utf8') as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            n += 1

    print(f'Number of re-aligned files: {n}')


if __name__ == '__main__':
    paths = [
        'data/train/',
        'data/validate/',
        'data/test/',
    ]

    realign(paths)
