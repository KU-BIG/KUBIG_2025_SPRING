from models import *
from torch.utils.data import DataLoader
import json
from data_utils import JsonDataset, CollateWrapper, BLEUDataset
import time
from utils import dict_to_device, translate_test, load_sample_sentences, breakpoint, bleu_test, UnfreezeScheduler, \
    init_optimizer
import numpy as np
import pickle
import os


def train(model, param_norm, data, opt, epoch, print_every, device, unfreeze_scheduler, accumulate=3):
    try:
        accumulate = int(accumulate)
    except ValueError as e:
        print("accumulate cannot be cast to type int: {}".format(e))
    assert accumulate > 0, "accumulate should be an integer greater than zero."
    model.train()
    losses = []
    duration_b = 1
    opt.zero_grad(set_to_none=True)
    for mb, d in enumerate(data):
        batch = int(mb / accumulate)
        d = dict_to_device(d, device)
        try:
            loss = model.loss(d, duration_b=duration_b) / accumulate
        except Exception as e:
            print(e)
            continue
        loss.backward()
        losses.append(loss.item())
        if (mb % accumulate == 0 and mb + 1 >= accumulate) or (mb == len(data) - 1):
            norm_loss = param_norm()
            norm_loss.backward()
            opt.step()
            unfreeze_scheduler.step()
            opt.zero_grad(set_to_none=True)
            if batch % print_every == 0:
                print('epoch: {}, batch: {}/{}, loss: {}'.format(
                    epoch,
                    batch,
                    len(data) // accumulate,
                    np.mean(losses[-accumulate:])
                ))
    return losses


@torch.no_grad()
def test(model, data, print_every, device, accumulate):
    model.eval()
    type_scores = []
    sign_scores = []
    timing_scores = []
    pointing_scores = []
    val_scores = []
    for mb, d in enumerate(data):
        d = dict_to_device(d, device)
        loss = model.loss(
            d,
            return_mean=False,
            # mean_weights=[1, 1, 2],
            testing=True,
        )
        type_scores.append(loss['type'].item() / accumulate)
        sign_scores.append(loss['token'].item() / accumulate)
        timing_scores.append(loss['duration'].item() / accumulate)
        if 'pointing' in loss.keys():
            pointing_scores.append(loss['pointing'].item() / accumulate)
        mn = [
            type_scores[-1],
            sign_scores[-1],
            timing_scores[-1],
        ] + (pointing_scores[-1:] if len(pointing_scores) > 0 else [])
        val_scores.append(
            np.mean(mn)
        )
        if mb % print_every == 0:
            print('Validating-> mb: {}, mode score: {}, gloss score: {}, timing score: {}'.format(
                mb,
                sum(type_scores)/(mb+1),
                sum(sign_scores)/(mb+1),
                sum(timing_scores)/(mb+1),
                sum(pointing_scores)/(mb+1)
            ), end='\r')

    val_score = np.mean(val_scores)
    type_score = np.mean(type_scores)
    sign_score = np.mean(sign_scores)
    timing_score = np.mean(timing_scores)
    pointing_score = np.mean(pointing_scores if len(pointing_scores) > 0 else [0])

    print('Val loss: {}, (mean of mode: {}, gloss: {}, timing: {}, pointing: {})\n\n'.format(
        val_score,
        type_score,
        sign_score,
        timing_score,
        pointing_score,
    ))
    return val_score, type_score, sign_score, timing_score, pointing_score,


def save_losses(d, path):
    with open(path, 'wb') as f:
        pickle.dump(d, f)


def main():
    """
    Main training script.
    """
    with open('./config.json', 'rb') as f:
        params = json.load(f)

    with open(params['ksl_vocab_path'], 'rb') as f:
        vocab = json.load(f)

    sign_vocab = list(set(vocab['sign_vocab']))
    nms_vocab = list(set(vocab['nms_vocab']))
    device = params['train']['device']
    model = Translator(
        sign_vocab=sign_vocab,
        nms_vocab=nms_vocab,
        max_source_length=params['max_source_length'],
        max_target_length=params['max_target_length'],
        ksl_embedding_method=params['ksl_embedding_method'],
        ksl_embedding_split=params['ksl_embedding_split'],
        ksl_embedding_init_pretrained=params['train']['ksl_embedding_init_pretrained'],
        channel_to_gloss=params['channel_to_gloss'],
        animation_mapping_path=params['animation_mapping_path'],
        ignore_nms=params['ignore_nms'],
        device=device,
    ).to(device)

    if params['train']['load_model'] and params['train']['load_id'] != '':
        model.load_state_dict(torch.load('{}{}_model.pt'.format(
            params['train']['model_path'],
            params['train']['load_id'],
        )))
    model.to(params['train']['device'])

    cw = CollateWrapper(model)

    if params['use_dataset_pickles']:
        train_pickle_path = os.path.join(params['pickle_directory'], params['train_pickle_name'])
        val_pickle_path = os.path.join(params['pickle_directory'], params['val_pickle_name'])
        bleu_val_pickle_path = os.path.join(params['pickle_directory'], params['bleu_val_pickle_name'])
        with open(train_pickle_path, 'rb') as f:
            dataset_train = pickle.load(f)
        dataset_train.model = model
        with open(val_pickle_path, 'rb') as f:
            dataset_val = pickle.load(f)
        dataset_val.model = model
        with open(bleu_val_pickle_path, 'rb') as f:
            dataset_bleu_val = pickle.load(f)
    else:
        dataset_train = JsonDataset(path=params['data_train_path'], model=model, preload=params['preload_data'])
        dataset_val = JsonDataset(path=params['data_validate_path'], model=model, preload=params['preload_data'])
        dataset_bleu_val = BLEUDataset(path=params['data_validate_path'], model=model)

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=params['train']['batch_size'],
        collate_fn=cw.collate,
        shuffle=True,
        num_workers=params['train']['n_dl_workers'],
    )
    dataloader_val = DataLoader(
        dataset=dataset_val,
        batch_size=params['train']['val_batch_size'],
        collate_fn=cw.collate,
        shuffle=True,
        num_workers=params['train']['val_n_dl_workers'],
    )
    dataloader_bleu_val = DataLoader(
        dataset=dataset_bleu_val,
        batch_size=params['train']['val_batch_size'],
        collate_fn=dataset_bleu_val.collate,
        shuffle=False,
        num_workers=params['train']['val_n_dl_workers'],
    )

    new_parameters = [
        {'params': model.ksl_wte.parameters(), 'lr': params['train']['sign_lr']},
        {'params': model.ksl_wte_alt.parameters(), 'lr': params['train']['sign_lr']},
        {'params': model.type_wte_alt.parameters(), 'lr': params['train']['sign_lr']},
        {'params': model.ksl_embedding.parameters(), 'lr': params['train']['sign_lr']},
        {'params': model.switch.parameters(), 'lr': params['train']['sign_lr']},
        {'params': model.ksl_head.parameters(), 'lr': params['train']['sign_lr']},
        {'params': model.channel_tokens, 'lr': params['train']['sign_lr']},
        {'params': model.duration_head.parameters(), 'lr': params['train']['timing_lr']},
        {'params': model.pointing_attention.parameters(), 'lr': params['train']['point_lr']},
        {'params': model.pointing_head.parameters(), 'lr': params['train']['point_lr']},
    ]
    param_norm = L2Model(*new_parameters)
    add_params = tuple()
    freeze_strategy = params['train']['freeze_strategies'][params['train']['freeze_pointer']]
    optimizer_name = params['train']['optimizers'][params['train']['optimizer_pointer']]
    load_optimizer = params['train']['load_optimizer'] and params['train']['load_id'] != ''
    if load_optimizer:
        if freeze_strategy == 'none':
            optimizer = init_optimizer(
                optimizer=optimizer_name,
                params=[{'params': model.model.parameters(), 'lr': params['train']['model_lr']}] + new_parameters,
                lr=params['train']['model_lr']
            )
        elif freeze_strategy == 'pretrained':
            optimizer = init_optimizer(
                optimizer=optimizer_name,
                params=new_parameters,
                lr=params['train']['model_lr'],
            )
        elif freeze_strategy == 'attention':
            # to reduce overfitting, freeze attention and FC blocks in decoder layers.
            optimizer = init_optimizer(
                optimizer=optimizer_name,
                params=[
                    {'params': block.ln_1.parameters(), 'lr': params['train']['model_lr']} for
                    block in model.model.h
                ] + [
                    {'params': block.ln_2.parameters(), 'lr': params['train']['model_lr']} for
                    block in model.model.h
                ] + new_parameters,
                lr=params['train']['model_lr'],
            )
        else:
            raise ValueError

        optimizer.load_state_dict(torch.load('{}{}_optimizer.pt'.format(
            params['train']['model_path'],
            params['train']['load_id'],
        )))
    else:
        optimizer = init_optimizer(
            optimizer=optimizer_name,
            params=new_parameters,
            lr=params['train']['model_lr'],
        )
        if freeze_strategy == 'none':
            add_params = ({'params': model.model.parameters(), 'lr': params['train']['model_lr']}, )
        elif freeze_strategy == 'attention':
            add_params = tuple()
            for block in model.model.h:
                add_params += (
                    {'params': block.ln_1.parameters(), 'lr': params['train']['model_lr']},
                    {'params': block.ln_2.parameters(), 'lr': params['train']['model_lr']},
                )
    unfreeze_step = params['train']['unfreeze_step']
    add_params = [(item, unfreeze_step) for item in add_params]  # give same unfreeze step for now.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, factor=.5, patience=params['train']['patience']
    )
    unfreeze_scheduler = UnfreezeScheduler(
        optimizer=optimizer,
        plateau_scheduler=scheduler,
        param_norm=param_norm,
        param_steps=add_params,
    )

    test_sentences = load_sample_sentences(params['data_validate_path'])

    losses = []
    val_scores = []
    type_scores = []
    sign_scores = []
    timing_scores = []
    pointing_scores = []
    bleu_score = []
    best_val_score = float('inf') 
    for epoch in range(params['train']['epochs']):
        train_start = time.time()
        ls = train(
            model=model,
            param_norm=param_norm,
            data=dataloader_train,
            opt=optimizer,
            epoch=epoch,
            print_every=params['train']['print_every'],
            device=device,
            unfreeze_scheduler=unfreeze_scheduler,
            accumulate=params['train']['accumulate'],
        )
        print('Epoch train time: {} seconds\n'.format(time.time() - train_start))
        losses.append(ls)

        if epoch % params['train']['eval_every'] == 0:
            vs, tys, ss, tis, ps = test(
                model=model,
                data=dataloader_val,
                print_every=params['train']['print_every'],
                device=device,
                accumulate=params['train']['accumulate'],
            )
            val_scores.append(vs)
            type_scores.append(tys)
            sign_scores.append(ss)
            timing_scores.append(tis)
            pointing_scores.append(ps)
            scheduler.step(vs)
            if vs < best_val_score:
                best_val_score = vs
                print(f"Saving new best model at epoch {epoch} with val loss {vs:.4f}")
                model.cpu()
                torch.save(model.state_dict(), f"{params['train']['model_path']}{params['train']['experiment_id']}_best_model.pt")
                model.to(device)
        if epoch > 0 and (epoch % params['train']['bleu_every'] == 0 or epoch == params['train']['epochs'] - 1):
            bleu_score.append(bleu_test(
                model,
                data=dataloader_bleu_val,
                device=device,
                n_beams=params['train']['n_beams'],
                top_k=params['train']['top_k'],
                top_p=params['train']['top_p'],
                ngram=4,
            ))
            print(f'BLEU score: {bleu_score[-1]}\n\n')

        if epoch % params['train']['save_model_every'] == 0 or epoch == params['train']['epochs'] - 1:
            model.cpu()
            torch.save(model.state_dict(), f"{params['train']['model_path']}{params['train']['experiment_id']}_epoch{epoch}.pt")

            model.to(device)
            torch.save(optimizer.state_dict(), f"{params['train']['model_path']}{params['train']['experiment_id']}_epoch{epoch}_optimizer.pt")


        save_losses(
            d={
                'train_loss': losses,
                'val_loss': val_scores,
                'val_type_loss': type_scores,
                'val_sign_loss': sign_scores,
                'val_timing_loss': timing_scores,
                'val_pointing_loss': pointing_scores,
                'bleu_score': bleu_score,
            },
            path='{}{}_metrics.pkl'.format(params['train']['model_path'], params['train']['experiment_id']),
        )

    print('\nTrained model translation: \n')
    print(test_sentences)
    translate_test(model, test_sentences=test_sentences, device=device, print_translation=True)

    model.cpu()
    torch.save(model.state_dict(), f"{params['train']['model_path']}{params['train']['experiment_id']}_epoch{params['train']['epochs']}.pt")
    torch.save(optimizer.state_dict(), f"{params['train']['model_path']}{params['train']['experiment_id']}_epoch{params['train']['epochs']}_optimizer.pt")


    print("\n================= TEST SENTENCE TRANSLATIONS =================")

    translations = model.batch_translate(
        sequences=test_sentences,
        device=device,
        player_script=False,
        nia_data_script=False,
        top_k=5,
        top_p=None,
        n_beams=1,
    )

    for i, sentence in enumerate(test_sentences):
        print(f"\n📝 Input: {sentence}")
        pred = translations['list'][i]
        if not pred:
            print("⚠️  No gloss predicted.")
            continue
        for gloss in pred:
            print(f"  🔤 [{gloss['mode']}] {gloss['gloss']} ({gloss['duration']:.2f}s)")

if __name__ == '__main__':
    main()

