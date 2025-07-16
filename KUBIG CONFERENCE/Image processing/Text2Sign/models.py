import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer, PreTrainedTokenizerFast
from transformers.cache_utils import DynamicCache
from pandas import Series
from typing import List, Union, Optional, Dict
import numpy as np
from utils import print_dict
import copy
from collections import defaultdict
import json


class L2Model(nn.Module):
    def __init__(self, *params):
        super(L2Model, self).__init__()
        self.params = []
        self.add_params(*params)

    def add_params(self, *params):
        self.params.extend(params)

    def norm(self, x, eps=0.00001):
        return x.pow(2).mean().clamp(eps).pow(0.5)

    def forward(self):
        norm = 0
        c = 0
        for param_group in self.params:
            for param in param_group['params']:
                norm += self.norm(param)
                c += 1
        return norm / c


class Translator(nn.Module):
    """
    """
    def __init__(self,
                 sign_vocab: List[str],
                 nms_vocab: List[str],
                 max_source_length: Optional[int] = 128,
                 max_target_length: Optional[int] = 128,
                 ksl_embedding_dim: Optional[int] = 128,
                 ksl_embedding_method: Optional[str] = 'linear',
                 ksl_embedding_split: Optional[float] = 0.5,  # only used if ksl_embedding_method is 'concat'
                 ksl_embedding_init_pretrained: Optional[bool] = False,
                 channel_to_gloss: Optional[bool] = False,
                 animation_mapping_path: Optional[str] = None,
                 ignore_nms: Optional[bool] = False,
                 device: Optional[str] = 'cpu',
                 ):
        """
        Model for translating Korean text (한글) into Korean sign language glosses (한국수어).

        Currently 수지 and 비수지 are predicted from a shared head (with a mask over decoding dictionaries so that 수지 are
        predicted first (by masking all 비수지 tokens when decoding) and then 비수지 are predicted (by masking all 수지
        tokens).

        :param sign_vocab: List of KSL types
        :type sign_vocab: list
        :param nms_vocab: List of KSL Non-manual signs
        :type nms_vocab: list
        :param max_source_length: Maximum sequence length (after tokenization) of source sentence, defaults to 128
        :type max_source_length: int, optional
        :param max_target_length: Maximum length of target sequence, defaults to 128
        :type max_target_length: int, optional
        :param ksl_embedding_dim: Dimension of embedding weight matrix for KSL tokens, defaults to 128. Only used if
            ksl_embedding_method is "linear"
        :type ksl_embedding_dim: int
        :param ksl_embedding_method: Method for embedding KSL tokens using both type and channel information. Options
            are: "linear", "concat", and "sum". Defaults to "linear". If "linear", each KSL token's learned embedding is
            concated with a channel identifier and then sent through a linear layer. If "concat", each KSL token's
            learned embedding is concated with a channel identifier's learned embedding and used directly. If "sum",
            each KSL token's learned embedding is summed with its channel's learned embedding.
        :type ksl_embedding_method: str
        :param ksl_embedding_split: Ratio of hidden dimension to use for KSL token's embedding vs for channel's
            embedding (only used if ksl_embedding_method is "concat"), defaults to 0.5
        :type ksl_embedding_split: float
        :param ksl_embedding_init_pretrained: If true, initializes ksl embedding for special tokens from pretrained
            weights, defaults to False. Only used if ksl_embedding_method is not "linear"
        :type ksl_embedding_init_pretrained: bool
        :param channel_to_gloss:
        :type channel_to_gloss: bool
        :param animation_mapping_path: Path to gloss -> animation mapping file
        :type animation_mapping_path: str
        :param device: Device to train model on, defaults to 'cpu'
        :param ignore_nms: Flag to ignore NMS in data preprocessing
        :type ignore_nms: bool
        :type device: str or torch.device, optional
        :return: Object of type Translator (inherits from torch.nn.Module)
        """
        super(Translator, self).__init__()
        self.source_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>',
            mask_token='<mask>'
        )
        self.model = GPT2Model.from_pretrained('skt/kogpt2-base-v2')
        n_embd = self.model.wte.embedding_dim

        # define special tokens here
        special_tokens = [
            "UNK",
            "PAD",
            "BOS",
            "EOS",
            "STEP",
        ]
        special_types = [
            'BOS',
            'EOS',
            'BOTH',
            'STRONG',
            'WEAK',
            'NMS',
            'FS_STRONG',
            'FS_WEAK',
            'STEP',
        ]
        self.special_tokens = {special_tokens[i]: i for i in range(len(special_tokens))}
        self.special_tokens_rev = {i: special_tokens[i] for i in range(len(special_tokens))}
        self.special_types = {special_types[i]: i for i in range(len(special_types))}
        self.special_types_rev = {i: special_types[i] for i in range(len(special_types))}

        sign_vocab = sorted(sign_vocab)
        nms_vocab = sorted(nms_vocab)

        extended_vocab = special_tokens + sign_vocab + nms_vocab
        # mask everything except 수지
        self.sign_mask = list(range(len(special_tokens))) + \
                         list(range(len(special_tokens) + len(sign_vocab), len(extended_vocab)))
        # mask everything except 비수지
        self.nms_mask = list(range(len(special_tokens) + len(sign_vocab)))

        self.ksl_token_series = Series(data=range(len(extended_vocab)), index=extended_vocab)
        self.token_ksl_series = Series(data=extended_vocab, index=range(len(extended_vocab)))

        self.ksl_embedding_method = ksl_embedding_method
        self.ksl_wte = nn.Embedding(num_embeddings=len(extended_vocab), embedding_dim=ksl_embedding_dim)
        # self.nms_wte = nn.Embedding(num_embeddings=len(nms_vocab), embedding_dim=ksl_embedding_dim)
        self.ksl_wte_alt = nn.Embedding(
            num_embeddings=len(extended_vocab),
            embedding_dim=int(np.floor(n_embd * ksl_embedding_split)) if ksl_embedding_method == 'concat' else n_embd
        )
        self.type_wte_alt = nn.Embedding(
            num_embeddings=len(special_types),
            embedding_dim=int(np.ceil(n_embd * (1-ksl_embedding_split))) if ksl_embedding_method == 'concat' else n_embd
        )
        # add pretrained BOS and EOS
        if ksl_embedding_init_pretrained and ksl_embedding_method != 'linear':  # does not help much.
            with torch.no_grad():
                self.ksl_wte_alt.weight[self.ksl_token_series['BOS']] = \
                    self.model.wte.weight[1, :self.ksl_wte_alt.weight.shape[-1]]  # kogpt uses 1 for both BOS and EOS
                self.ksl_wte_alt.weight[self.ksl_token_series['EOS']] = \
                    self.model.wte.weight[1, :self.ksl_wte_alt.weight.shape[-1]]
                self.type_wte_alt.weight[self.special_types['BOS']] = \
                    self.model.wte.weight[1, self.ksl_wte_alt.weight.shape[-1]:]
                self.type_wte_alt.weight[self.special_types['EOS']] = \
                    self.model.wte.weight[1, self.ksl_wte_alt.weight.shape[-1]:]

        self.ksl_embedding = nn.Linear(ksl_embedding_dim + 1, n_embd)  # + 2 if including durations in embedding
        self.hidden_size = n_embd

        self.switch = nn.Linear(n_embd, len(special_types), bias=False)  # BOS/EOS, 수지(BOTH, STRONG, WEAK), 비수지, STEP

        self.channel_to_gloss = channel_to_gloss  # flag to use channel and hidden space to calculate gloss id
        self.ksl_head = nn.Sequential(
            nn.Linear(n_embd + (3 if channel_to_gloss else 0), 2 * n_embd),
            nn.LeakyReLU(),
            nn.Linear(2 * n_embd, len(extended_vocab), bias=False),
        )
        self.channel_tokens = nn.Parameter(torch.ones(9, 3))  # only trained if channel_to_gloss == True

        self.duration_head = nn.Sequential(
            nn.Linear(n_embd, int(n_embd / 2)),
            nn.Sigmoid(),
            nn.Linear(int(n_embd / 2), 1),
        )

        self.duration_mod = F.softplus  # (input, self.beta, self.threshold)

        self.pointing_attention = nn.MultiheadAttention(n_embd, num_heads=3)
        self.pointing_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(n_embd, 1)
        )

        self.position_head = nn.Linear(n_embd, 36)  # 9 Strong 1, 9 Strong 2, 9 Weak 1, 9 Weak 2

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.device = device

        self.postprocess_print = False

        self.animation_mapping = Series([])
        if animation_mapping_path is not None:
            try:
                with open(animation_mapping_path, 'rb') as f:
                    self.animation_mapping = Series(json.load(f))

            except Exception as e:
                print(f"Unable to locate animation mapping file. Defaulting to None.\nError: {e}")

        self.ignore_nms = ignore_nms

    def gloss_to_anim(self, gs: List[str]):
        """Returns an animation ID for each gloss or -1 if no matching animation ID found"""
        return self.animation_mapping.get(gs, default='-1')

    def dict_ksl_to_token(self, seq):
        """Returns a token for each gloss in seq."""
        return self.ksl_token_series.get(seq, default=self.special_tokens['UNK'])

    def dict_token_to_ksl(self, seq):
        """Returns a KSL gloss for each token in seq."""
        return self.token_ksl_series.get(seq, default='UNK')

    def trim(self, xs: list, max_len: int):
        """
        Returns a continuous subset of xs of length min(len(xs), max_len)

        :param xs: A list
        :type xs: list
        :param max_len: The maximum length to which to trim xs
        :type max_len: int
        :return: A continuous subset of xs of length min(len(xs), max_len)
        """
        if len(xs) <= max_len:
            return xs
        else:
            delta = len(xs) - max_len
            start = torch.randint(0, delta, size=(1,))
            return xs[start:start+max_len]

    def process_instance(
            self,
            data,
            ignore_nms: Optional[bool] = None,
    ):
        if ignore_nms is None:
            ignore_nms = self.ignore_nms

        source_text = data['korean_text']
        source_tokens = self.source_tokenizer.encode_plus(data['korean_text'], return_offsets_mapping=True)
        source_offsets = [tpl for tpl in source_tokens['offset_mapping'] if tpl != (0, 0)]
        source_tokens = source_tokens['input_ids'][:self.max_source_length]

        # process sign script
        sign_type = {
            'sign_gestures_both': 'BOTH',
            'sign_gestures_strong': 'STRONG',
            'sign_gestures_weak': 'WEAK',
        }
        fs_sign_type = {
            'sign_gestures_both': 'BOTH',
            'sign_gestures_strong': 'FS_STRONG',
            'sign_gestures_weak': 'FS_WEAK',
        }
        sign_type_targets = [
            'BOTH',
            'STRONG',
            'WEAK',
            'FS_STRONG',
            'FS_WEAK',
        ]
        sign_script = []
        channel_script = {
            'BOTH': [],
            'STRONG': [],
            'WEAK': [],
        }
        for key, ls in data['sign_script'].items():
            for item in ls:
                # for now, parse d and n with s
                if item['express'] == 'f' or item['gloss_id'][:2] in ['n:', 'x:']:
                    item_type = fs_sign_type[key]
                else:
                    item_type = sign_type[key]

                sign_script.append({
                    'type': item_type,
                    'token': item['gloss_id'],
                    'start': item['start'],
                    'duration': item['end'] - item['start'],
                    'other': item['sentence_loc'],
                })
                channel_script[item_type.replace('FS_', '')].append(
                    ('' if item['express'] != 'f' else 'f:') + item['gloss_id']
                )

        # process nms script
        nms_script = []
        if not ignore_nms:
            for key, ls in data['nms_script'].items():
                for item in ls:
                    nms_script.append({
                        'type': 'NMS',
                        'token': key,
                        'start': item['start'],
                        'duration': item['end'] - item['start'],
                        'other': item['descriptor'] if key == 'Mmo' else '',
                    })

        # combine and sort by start time
        script_ = sign_script + nms_script
        script_ = sorted(script_, key=lambda x: x['start'])

        # add step tokens
        previous_time = 0
        script = [{'type': 'BOS', 'token': 'BOS', 'start': 0, 'duration': 0}]
        for item in script_:
            if item['start'] != previous_time:
                step_item = {
                    'type': 'STEP',
                    'token': 'STEP',
                    'duration': item['start'] - previous_time
                }
                script.append(step_item)
            script.append(item)
            previous_time = item['start']
        script = script[:self.max_target_length]
        if script[-1]['type'] == 'STEP':
            script = script[:-1]
        script.append({'type': 'EOS', 'token': 'EOS', 'start': script[-1]['start'], 'duration': 0})

        gloss_list = [s['token'] for s in script if s['token'] not in ['STEP', 'UNK', 'BOS', 'EOS']]

        point_temp = [
            (s_i, s['other']) for s_i, s in enumerate(script)
            if s['type'] in sign_type_targets and s.get('other', {'start': ''})['start'] != ''
        ]

        # skip all (0, 0) pointing targets for now
        point_temp = [item for item in point_temp if (item[1]['start'], item[1]['end']) != (0, 0)]

        if len(point_temp) != 0:
            point_inds, point = zip(*[(item[0], (item[1]['start'], item[1]['end'])) for item in point_temp])
            point = list(point)
        else:
            point_inds = []
            point = []

        for pt_i in range(len(point)):
            pt_start = None
            pt_end = None
            for offset_i, offset in enumerate(source_offsets):
                if pt_start is None and point[pt_i][0] < offset[1]:
                    pt_start = offset_i
                if point[pt_i][1] < offset[0]:
                    pt_end = offset_i - 1
                    break
            if pt_end is None:
                pt_end = offset_i

            point[pt_i] = range(pt_start, pt_end)  # now in terms of token position instead of character position

        for s in script:
            s['token'] = self.ksl_token_series.get(s['token'], self.special_tokens['UNK'])

        # organize by target type
        types = [self.special_types[item['type']] for item in script]
        tokens = [item['token'] for item in script]
        signs, sign_inds = list(zip(*[
            (item['token'], i) for i, item in enumerate(script) if item['type'] in ['BOTH', 'STRONG', 'WEAK']
        ]))

        # empty list handling for NMS. All other lists should always be nonempty.
        nms_temp = [(item['token'], i) for i, item in enumerate(script) if item['type'] == 'NMS']
        nmss, nms_inds = list(zip(*nms_temp)) if nms_temp != [] else ([], [])

        durations = [item['duration'] for item in script]
        token_durations, token_inds = list(zip(*[
            (item['duration'], i) for i, item in enumerate(script) if item['type'] in ['BOTH', 'STRONG', 'WEAK', 'NMS']
        ]))

        step_durations, step_inds = list(zip(*[
            (item['duration'], i) for i, item in enumerate(script) if item['type'] == 'STEP'
        ]))

        output = {
            'source_text': source_text,
            'gloss_list': gloss_list,
            'source_tokens': source_tokens,
            'predict_from': len(source_tokens),
            'types': types,
            'tokens': tokens,
            'signs': signs,
            'sign_inds': sign_inds,
            'nmss': nmss,
            'nms_inds': nms_inds,
            'token_durations': token_durations,
            'token_inds': token_inds,
            'durations': durations,
            'step_durations': step_durations,
            'step_inds': step_inds,
            'point_inds': point_inds,
            'point': point,
            'script': channel_script,
        }

        return output

    def embed(
            self,
            data,
    ):
        device = data['source'].device
        embedding = torch.zeros(data['source'].shape[0], data['max_source'] + data['max_target'], self.hidden_size,
                                device=device
                                ).float()

        # need to backprop gradient to these calculations, so not handled in collate function
        embedding[data['source_inds'][0], data['source_inds'][1], :] = self.model.wte(data['source'])[data['source_inds']]

        if self.ksl_embedding_method == 'linear':  # embedding using linear layer
            embedding[data['type_inds'][0], data['type_inds'][1] + data['target_offset']] = self.ksl_embedding(
                torch.cat([
                    self.ksl_wte(data['tokens']),
                    data['types'].unsqueeze(-1),
                    # data['durations'].unsqueeze(-1),
                ], dim=-1)
            )
        elif self.ksl_embedding_method == 'concat': # embedding using dicts and then concating
            embedding[data['type_inds'][0], data['type_inds'][1] + data['target_offset']] = torch.cat([
                self.ksl_wte_alt(data['tokens']),
                self.type_wte_alt(data['types'].unsqueeze(-1)).squeeze(1),
            ], dim=-1)
        elif self.ksl_embedding_method == 'sum': # embedding using dicts and then summing
            embedding[data['type_inds'][0], data['type_inds'][1] + data['target_offset']] = \
                self.ksl_wte_alt(data['tokens']) + self.type_wte_alt(data['types'].unsqueeze(-1)).squeeze(1)
        else:
            assert ValueError, f'Specified "ksl_embedding_method" {self.ksl_embedding_method} is not recognized'
        return embedding

    def point(
            self,
            source_embeds: torch.Tensor,
            target_embeds_list: List[torch.Tensor],
            mask: torch.Tensor = None,
            fs_inds: List[torch.Tensor] = None,
            argmax: bool = True,
    ):
        # 디바이스 정보 확보
        device = source_embeds.device

        if fs_inds is not None:
            if isinstance(fs_inds[0], list):
                fs_inds = (torch.tensor(fs_inds[0]), fs_inds[1])

            # 모든 fs_inds를 디바이스에 올림
            fs_inds = (
                fs_inds[0].to(device),
                fs_inds[1].to(device)
            )

        output = []

        for col, target_embeds in enumerate(target_embeds_list):
            output_entry = None  # 기본값

            if fs_inds is None:
                inds = torch.arange(source_embeds.size(0), device=device)
            else:
                temp = torch.where(fs_inds[1] == col)[0]
                if temp.numel() == 0:
                    output.append(output_entry)
                    continue
                inds = fs_inds[0][temp]
                if inds.numel() == 0:
                    output.append(output_entry)
                    continue

            source_sel = source_embeds[inds]
            target_sel = target_embeds[inds]

            if source_sel.size(0) == 0 or target_sel.size(0) == 0:
                output.append(output_entry)
                continue

            try:
                combined = torch.cat([source_sel, target_sel], dim=1).permute(1, 0, 2)
            except Exception as e:
                print(f"[Error] Failed to concatenate tensors for column {col}: {e}")
                output.append(output_entry)
                continue

            combined = self.pointing_attention(combined, combined, combined)[0].permute(1, 0, 2)
            combined = self.pointing_head(combined)[:, :-1]

            if argmax:
                combined = combined.argmax(dim=1)

            output_entry = combined
            output.append(output_entry)

        return output


    def forward(
            self,
            input_ids=None,
            inputs_embeds=None,
            padding_mask=None,
            testing=False,
            duration_b=20,
            fs_inds=None,
            target_types=None,
    ):
        # currently just computes forward pass through every head for every token.
        pred = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=padding_mask,
            position_ids=None,
        )['last_hidden_state']

        pred_type = self.switch(pred)

        if self.channel_to_gloss:
            ksl_head_input = torch.cat([
                    pred,
                    self.channel_tokens[target_types if target_types is not None else pred_type.argmax(dim=-1)],
                ], dim=-1)
        else:
            ksl_head_input = pred
        pred_ksl = self.ksl_head(ksl_head_input)

        pred_duration = self.duration_mod(
            self.duration_head(
                pred.view(-1, pred.shape[-1])
            ), duration_b
        ).view(pred.shape[0], -1, 1)  # double check these dimensions

        # need optional indices to reduce size of this (where type == FS_STRONG or FS_WEAK)
        # pred_mod = pred[fs_inds] if fs_inds is not None else pred
        pred_point = self.point(
            source_embeds=inputs_embeds,
            target_embeds_list=[pred[:, col].unsqueeze(1) for col in range(pred.shape[1])],
            # mask: torch.Tensor = None,
            fs_inds=fs_inds,
            argmax=False,
        )

        output = {
            'type': pred_type,
            'ksl': pred_ksl,
            'duration': pred_duration,
            'point': pred_point,
        }

        return output

    def loss(
            self,
            data,
            return_mean=True,
            mean_weights=None,
            testing=False,
            duration_b=20,
    ):
        """
        Calculates loss. Input parameters match the outputs of self.make_training_array exactly.

        :param data:
        :param return_mean: If True, then all losses are averaged, else they are returned separately, defaults to True
        :type return_mean: bool, optional
        :param mean_weights:
        :return:
        """
        embedding = self.embed(data)

        fs_inds = (data['point_inds'][0], data['point_inds'][1].long())# + data['ksl_offset'] - 1)
        prediction = self.forward(
            inputs_embeds=embedding,
            padding_mask=data['padding_mask'],
            testing=testing,
            duration_b=duration_b,
            fs_inds=fs_inds,
            # target_types=,
        )

        type_inds = (data['type_inds_noeos'][0], data['type_inds_noeos'][1] + data['target_offset_noeos'])
        type_loss = F.cross_entropy(prediction['type'][type_inds], data['types_nobos'])

        ksl_inds = (data['ksl_inds'][0], data['ksl_inds'][1] + data['ksl_offset'] - 1)
        token_loss = F.cross_entropy(prediction['ksl'][ksl_inds], data['ksl'])

        duration_loss = F.mse_loss(
            prediction['duration'][type_inds].squeeze(),
            data['durations_nobos'],
        )

        # Unlike above, moved tensor slice to inside self.point to save memory
        losses = {
            'type': type_loss,
            'token': token_loss,
            'duration': duration_loss,
            # 'pointing': point_loss,
            # 'position': None,
        }

        valid_points = [p for p in prediction['point'] if p is not None]
        if len(valid_points) > 0:
            point_loss = F.binary_cross_entropy_with_logits(
                torch.cat(valid_points, dim=0).squeeze(-1)[:, :data['point'].shape[1]],
                data['point'],
            )
            losses['pointing'] = point_loss

        if return_mean:
            if mean_weights is None:
                mean_weights = [1 for _ in range(len(losses.values()))]
            else:
                assert len(mean_weights) == len(losses.values())
            return sum([loss * mean_weights[i] for i, loss in enumerate(losses.values())]) / len(losses.values())
        return losses

    def masked_2d_argmax(self, x: torch.Tensor, mask: Union[np.ndarray, torch.Tensor, list], dim: Optional[int] = 1,
                         keepdims: Optional[bool] = False, return_softmax: Optional[bool] = False,
                         return_top: Optional[int] = 1
                         ):
        """
        Returns the argmax of x along dimension dim, ignoring values marked in mask. mask values of True are ignored.
        """
        ### check mask dimensions
        x[:, mask] = -float('Inf')
        if return_top == 1:
            output = {'argmax': x.argmax(dim=dim, keepdims=keepdims)}
            if return_softmax:
                output['softmax'] = torch.softmax(x, dim=dim).max(dim=dim)[0]
        else:
            output = {'argmax': x.argsort(dim=dim, descending=True)}
            if return_softmax:
                output['softmax'] = torch.softmax(x, dim=dim).sort(dim=dim, descending=True)[0]
        return output

    def masked_top_k(self, logits, k):
        ranking = logits.argsort(dim=-1, descending=True)[:, :k]
        ranking_rows = torch.tile(torch.arange(ranking.shape[0]).unsqueeze(1), dims=(1, k))
        ranking = ranking.sort(dim=-1)[0]
        logits_mod = logits[ranking_rows, ranking]
        return logits_mod, ranking_rows, ranking

    def p_cutoff(self, arr, p):
        assert p < 1
        inds = torch.argsort(arr, descending=True)
        # ref = 1 - p
        sm = 0
        include = []
        for i in inds:
            include.append(i)
            sm += arr[i]
            if sm > p:
                break
        return include

    def masked_top_p(self, logits, p, temperature: Optional[int] = 1):
        logits = torch.softmax(logits / temperature, dim=-1)
        sample_over = []
        for row in logits:
            sample_over.append(self.p_cutoff(row, p))
        return sample_over

    def sample_2d(self, logits, ranking_k, ranking_p, temperature: Optional[float] = 1):
        if ranking_p is None:  # use ranking_k, logits should be shape: (batch, k)
            logits = torch.softmax(logits / temperature, dim=-1)
            sample = torch.multinomial(logits, 1)
            result = [ranking_k[row_i, sample_i] for row_i, sample_i in enumerate(sample)]
        else:  # use ranking_p
            sample = [  # rewrite as a for loop
                torch.multinomial(
                    torch.softmax(
                        torch.tensor(logits[row_i][torch.tensor(ranking_p[row_i])]) / temperature,
                        0,
                    ),
                    1,
                )
                for row_i in range(len(logits))
            ]
            result = [ranking_p[row_i][sample_i] for row_i, sample_i in enumerate(sample)]
            if ranking_k is not None:
                result = [ranking_k[row_i, sample_i] for row_i, sample_i in enumerate(result)]
        return torch.tensor(result)

    def decode_gloss(self, logits, mask, dim: Optional[int] = -1, keepdims: Optional[bool] = False,
                     top_k: Optional[int] = None, top_p: Optional[float] = None, n_beams: Optional[int] = 1,
                     return_softmax: Optional[bool] = False,
                     ):
        if top_k is None and top_p is None:
            result = self.masked_2d_argmax(
                logits,
                mask=mask,
                dim=dim,
                keepdims=keepdims,
                return_softmax=return_softmax,
                return_top=n_beams,
            )
            output = {
                'token': result['argmax'][:, :n_beams],
            }
            if return_softmax:
                # dim_0 = torch.tensor([[i] * n_beams for i in range(len(logits))])
                # dim_1 = output['token'].reshape(-1)
                # output['probability'] = result['softmax'][dim_0, dim_1]

                # output['probability'] = result['softmax'][:, n_beams]

                # pre sort or optimize this
                output['probability'] = result['softmax'].sort(descending=True)[0][:, :n_beams]
        else:
            device = logits.device
            logits = logits.cpu()
            ranking_k = None
            ranking_p = None

            if mask is not None:
                logits[:, mask] = -float('Inf')

            if top_k is not None:
                logits, ranking_rows, ranking_k = self.masked_top_k(logits, k=top_k)
            if top_p is not None:
                ranking_p = self.masked_top_p(logits, p=top_p)

            result = self.sample_2d(logits, ranking_k=ranking_k, ranking_p=ranking_p)
            result = result.to(device).unsqueeze(1)
            output = {'token': result}
            if return_softmax:
                output['probability'] = torch.ones_like(result)
        return output

    def embed_target(self, tokens, types, durations):
        if self.ksl_embedding_method == 'linear':  # embedding using linear layer
            out = self.ksl_embedding(
                torch.cat([
                    self.ksl_wte(tokens),
                    types,
                    # durations,
                ], dim=-1)
            )
        elif self.ksl_embedding_method == 'concat':  # embedding using dict and then concat
            out = torch.cat([
                self.ksl_wte_alt(tokens),
                self.type_wte_alt(types.squeeze(1)),
            ], dim=-1)
        elif self.ksl_embedding_method == 'sum':  # embedding using dict and then sum
            out = self.ksl_wte_alt(tokens) + self.type_wte_alt(types.squeeze(1))
        else:
            assert ValueError, f'Specified "ksl_embedding_method" {self.ksl_embedding_method} is not recognized'
        return out

    def decode_point(self, p, seq_input, seq_mask=None, dim=1, drop_firstlast=True, mask=None):
        """
        Decode pointer data (simplified version)
        """
        p.squeeze(-1)[(1 - seq_mask).bool()] = -float('Inf')
        point = p.argmax(dim=dim)
        point = point.to(seq_input.device)
        # mod = int(drop_firstlast)
        # return [self.source_tokenizer.decode(s[point[s_i] + mod]) for s_i, s in enumerate(seq_input)]
        return [self.source_tokenizer.decode(s[point[s_i]]) for s_i, s in enumerate(seq_input)]

    def formatter(self, script):
        """
        Format for animation player.
        """
        map = {
            'Tbt': 'SequenceMouth',
            'Mctr': 'SequenceMouth',
            'Mmo': 'SequenceMouth',
            'Mo1': 'SequenceFace',
            'Ci': 'SequenceFace',
            'EBf': 'SequenceFace',
            'Hs': 'SequenceFace',
            'Hno': 'SequenceFace',
        }
        d = {
            "SequenceHandBoth": script['sign_script']['sign_gestures_both'],
            "SequenceHandRight": script['sign_script']['sign_gestures_strong'],
            "SequenceHandLeft": script['sign_script']['sign_gestures_weak'],
            "SequenceBody": [],
            "SequenceFace": [],
            "SequenceMouth": [],
        }
        for k, vs in script['nms_script'].items():
            for v in vs:
                v['gloss'] = k
            d[map[k]].extend(vs)
        d['SequenceBody'] = sorted(d['SequenceBody'], key=lambda x: x['start'])
        d['SequenceFace'] = sorted(d['SequenceFace'], key=lambda x: x['start'])
        d['SequenceMouth'] = sorted(d['SequenceMouth'], key=lambda x: x['start'])

        for k, vs in d.items():
            for v in vs:
                v['BlockExtraData'] = []
                v['StartTime'] = v['start']
                v['EndTime'] = v['end']
                del v['start'], v['end']

                if k in ['SequenceHandBoth', 'SequenceHandRight', 'SequenceHandLeft']:
                    if v['express'] == 's':
                        v['BlockType'] = 'animation'
                        # print(v)
                        v['BlockValue'] = str(self.gloss_to_anim(v['gloss']))
                        v['BlockValueText'] = v['gloss']
                    elif v['express'] == 'f':
                        v['BlockType'] = 'fingerspell'
                        v['BlockValue'] = v['gloss'][2:]
                        v['BlockValueText'] = v['gloss'][2:]
                        # add parsing for numbers
                    else:
                        raise NotImplementedError
                    del v['gloss'], v['express']
                else:
                    v['BlockType'] = 'NonManual'
                    # v['BlockValue'] = self.gloss_to_anim(v['gloss'])
                    v['BlockValue'] = v['gloss']
                    v['BlockValueText'] = v['gloss']
                    del v['gloss']
        return d

    def add_neutral_pose(self, xs, delta=0.1):
        """
        Player requires first and last animations to be neutral pose with in SequenceHandBoth. This function adds this
        pose to the start and end of SequenceHandBoth and shifts all other tokens accordingly.
        """
        all_keys = [
            'SequenceHandBoth', 'SequenceHandRight', 'SequenceHandLeft', 'SequenceBody', 'SequenceFace',
            'SequenceMouth'
        ]

        # set start time
        start = delta
        for key in all_keys:
            if len(xs[key]) == 0:
                continue
            start = min(start, xs[key][0]['StartTime'])
        if start == 0:  # if start is at zero, push everything back delta seconds
            for key in all_keys:
                for item in xs[key]:
                    item['StartTime'] += delta
                    item['EndTime'] += delta
            start = delta

        # add neutral pose
        end = max([item['EndTime'] for items in xs.values() for item in items])
        neutral_0 = {
            "StartTime": 0,
            "EndTime": start,
            "BlockType": "animation",
            "BlockValue": "25209",
            "BlockValueText": "neutral",
            "BlockExtraData": [],
        }
        neutral_1 = {
            "StartTime": end,
            "EndTime": end + delta,
            "BlockType": "animation",
            "BlockValue": "25209",
            "BlockValueText": "neutral",
            "BlockExtraData": [],
        }
        xs['SequenceHandBoth'] = [neutral_0] + xs['SequenceHandBoth'] + [neutral_1]
        return xs

    def postprocess(self, xs, *, source_text='', nia_data_script=False, player_script=False):
        """Interpret pointing, calculate start stop, remove step (and UNK until finish pointing)."""
        overlap = {  # overlap based on player rules
            'BOTH': ['BOTH', 'STRONG', 'WEAK'],
            'STRONG': ['BOTH', 'STRONG'],
            'WEAK': ['BOTH', 'WEAK'],
            'NMS_Ci': ['NMS_Ci', 'NMS_EBf', 'NMS_Mo1', 'NMS_Hs', 'NMS_Hno'],
            'NMS_EBf': ['NMS_Ci', 'NMS_EBf', 'NMS_Mo1', 'NMS_Hs', 'NMS_Hno'],
            'NMS_Mo1': ['NMS_Ci', 'NMS_EBf', 'NMS_Mo1', 'NMS_Hs', 'NMS_Hno'],
            'NMS_Hno': ['NMS_Ci', 'NMS_EBf', 'NMS_Mo1', 'NMS_Hs', 'NMS_Hno'],
            'NMS_Mmo': ['NMS_Mmo', 'NMS_Mctr', 'NMS_Tbt'],
            'NMS_Hs': ['NMS_Ci', 'NMS_EBf', 'NMS_Mo1', 'NMS_Hs', 'NMS_Hno'],
            'NMS_Mctr': ['NMS_Mmo', 'NMS_Mctr', 'NMS_Tbt'],
            'NMS_Tbt': ['NMS_Mmo', 'NMS_Mctr', 'NMS_Tbt'],
            'STEP': ['STEP'],
        }

        script_map = {  # map keys between different formats
            'BOTH': ['sign_script', 'sign_gestures_both'],
            'STRONG': ['sign_script', 'sign_gestures_strong'],
            'WEAK': ['sign_script', 'sign_gestures_weak'],
            'NMS_Ci': ['nms_script', 'Ci'],
            'NMS_EBf': ['nms_script', 'EBf'],
            'NMS_Mo1': ['nms_script', 'Mo1'],
            'NMS_Hno': ['nms_script', 'Hno'],
            'NMS_Mmo': ['nms_script', 'Mmo'],
            'NMS_Hs': ['nms_script', 'Hs'],
            'NMS_Mctr': ['nms_script', 'Mctr'],
            'NMS_Tbt': ['nms_script', 'Tbt'],
        }

        channel = {  # track current time marker for each channel (will use max(channel, STEP))
            'BOTH': 0,
            'STRONG': 0,
            'WEAK': 0,
            'NMS_Ci': 0,
            'NMS_EBf': 0,
            'NMS_Mo1': 0,
            'NMS_Hno': 0,
            'NMS_Mmo': 0,
            'NMS_Hs': 0,
            'NMS_Mctr': 0,
            'NMS_Tbt': 0,
            'STEP': 0,
        }

        script = {
            'sign_script': {
                'sign_gestures_both': [],
                'sign_gestures_strong': [],
                'sign_gestures_weak': [],
            },
            'nms_script': {
                'Mmo': [],  # predict time but not mouth shape.
                'Hno': [],
                'Mo1': [],
                'Hs': [],
                'EBf': [],
                'Mctr': [],
                'Ci': [],
                'Tbt': [],
            }
        }
        for x in xs:
            cat = x['mode']
            if cat == 'NMS':
                cat = cat + '_' + x['gloss']
            cross_cats = overlap[cat]
            base = channel['STEP']
            for cc in cross_cats:
                base = max(base, channel[cc])
            channel['STEP'] = base  # if overlap, need to update cursor
            x['start'] = channel['STEP']
            x['end'] = x['start'] + x['duration']
            channel[cat] = channel['STEP'] + x['duration']
            if cat != 'STEP':
                level = script_map[cat]
                if level[0] == 'nms_script':
                    item = {'start': x['start'], 'end': x['end']}
                else:
                    express = 's' if x['gloss'][:2] != 'f:' else 'f'
                    item = {'gloss': x['gloss'], 'express': express, 'start': x['start'], 'end': x['end']}
                script[level[0]][level[1]].append(item)

        if self.postprocess_print:
            print_dict(script)
            self.postprocess_print = False

        output = {
            'list': xs,
        }
        if nia_data_script:
            output['nia_data_script'] = copy.deepcopy(script)
            output['nia_data_script']['korean_text'] = source_text
        if player_script:
            output['player_script'] = self.add_neutral_pose(self.formatter(copy.deepcopy(script)))
            output['player_script']['SourceText'] = source_text
        return output

    def build_past_keys(self, xs, eos):
        ref = np.where([not e for e in eos])[0]
        if ref.shape[0] > 0:
            ref_key = xs[ref[0]]
        keys = []
        for layer in range(12):
            key_layer = []
            for stack in range(2):
                tensors = [
                    ref_key[layer][stack] if e else x[layer][stack]
                    for x, e in zip(xs, eos)
                ]
                # 모든 텐서 크기를 맞추기 위한 padding
                max_len = max(t.shape[1] for t in tensors)
                padded = []
                for t in tensors:
                    if t.shape[1] < max_len:
                        pad_size = max_len - t.shape[1]
                        padding = torch.zeros(
                            t.shape[0], pad_size, t.shape[2],
                            device=t.device,
                            dtype=t.dtype
                        )
                        t = torch.cat([t, padding], dim=1)
                    elif t.shape[1] > max_len:
                        t = t[:, :max_len, :]
                    padded.append(t)
                key_layer.append(torch.stack(padded))
            keys.append((key_layer[0], key_layer[1]))
        return keys


    def extract_past_keys(self, keys, row):
        return [[stack[row] for stack in layer] for layer in keys]

    def beam_probability(self, xs):
        return np.prod([x ** 2 for x in xs]) ** 0.5

    @torch.no_grad()
    def batch_translate(
            self,
            sequences: Union[List[str], str],
            max_length: Optional[int] = None,
            nia_data_script: Optional[bool] = False,
            player_script: Optional[bool] = False,
            n_beams: Optional[int] = 5,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            device=None,
    ):
        """
        Simple implementation of greedy decoding with sampling. outputs are returned in a dict with keys: "list",
        "nia_data_script" (if nia_data_script == True), and "player_script" (if player_script == True).

        :param sequences: Input sequence(s) as a list of strings or (only if translating a single input) a single string
        :param max_length: Max translation length
        :param nia_data_script: Flag to output in project format
        :param player_script: Flag to output in player format
        :param n_beams: Number of beams to use in modified beam search
        :param top_k: k for top k sampling, default to None
        :param top_p: p for nucleus sampling, default to None
        :param device: Torch device
        """
        if device is None:
            device = self.device
        if max_length is None:
            max_length = self.max_target_length

        if type(sequences) == str:
            sequences = [sequences]
        else:
            assert isinstance(sequences, list) and isinstance(sequences[0], str)

        if n_beams is not None:
            if n_beams > 1 and (top_k is not None or top_p is not None):
                print("Sampling is disabled when n_beams > 1. Ignoring top_k and top_p")
                top_k = None
                top_p = None
        else:
            n_beams = 1

        fs_mode_map = {
            self.special_types['FS_STRONG']: 'STRONG',
            self.special_types['FS_WEAK']: 'WEAK',
        }

        n_seqs = len(sequences)
        seq = self.source_tokenizer(sequences, padding=True, return_tensors='pt')
        input_ids = seq['input_ids'].to(device)
        input_ids = self.model.wte(input_ids)
        position_ids = torch.stack([
            torch.tensor([row[:i].sum() for i in range(len(row))])
            for row in seq['attention_mask']
        ]).to(input_ids.device)

        next_token = self.embed_target(
            tokens=torch.tile(torch.tensor([self.special_tokens['BOS']], device=device), dims=(n_seqs, 1)),
            types=torch.tile(torch.tensor([self.special_types['BOS']], device=device), dims=(n_seqs, 1, 1)),
            durations=torch.zeros(n_seqs, 1, 1, device=device),
        )
        input_ids = torch.cat([input_ids, next_token], dim=1)
        position_ids = torch.cat([position_ids, position_ids[:, -1:] + 1], dim=1)
        next_token = input_ids
        attention_mask = torch.cat([seq['attention_mask'], torch.ones(n_seqs, 1)], dim=-1).to(device)

        i = 0

        data = [
            [{
                'embedding': token,
                'past_key_values': None,
                'position_ids': position_ids[row, -1],
                'probabilities': [],
                'prediction': [],
                'translation': [],
                'eos': False,
            }]
            for row, token in enumerate(next_token)
        ]

        while i < max_length:
            n_beams_eff = max([len(item) for item in data])
            for beam_i in range(n_beams_eff):
                data_slice = [item[beam_i] for item in data if len(item) >= (beam_i + 1)]
                eos = [item['eos'] for item in data_slice]
                if all(eos):
                    continue
                embedding = torch.stack([item['embedding'] for item in data_slice])
                past_key_values = None if i == 0 else self.build_past_keys([item['past_key_values'] for item in data_slice], eos)
                position_ids = position_ids if i == 0 else torch.stack([item['position_ids'] for item in data_slice])

                model_output = self.model(
                    inputs_embeds=embedding,
                    use_cache=True,
                    past_key_values=DynamicCache.from_legacy_cache(past_key_values),
                    attention_mask=attention_mask if i == 0 else None,
                    position_ids=position_ids,
                )

                enc = model_output['last_hidden_state'][:, -1]
                past_key_values = model_output['past_key_values']

                pred_output = self.masked_2d_argmax(  # ignore 0 (BOS)
                    self.switch(enc),
                    mask=[self.special_types['BOS']],
                    dim=-1,
                    return_softmax=True
                )
                pred_type = pred_output['argmax']
                pred_probs = torch.stack([pred_output['softmax']] * n_beams, dim=-1)

                pred_ksl = torch.zeros(len(pred_type), n_beams, device=device).long()
                pred_fs = [[] for _ in range(n_seqs)]

                # tokens 2,3,4 (수지)
                sign_here = torch.where(
                    pred_type.eq(self.special_types['BOTH']) |
                    pred_type.eq(self.special_types['STRONG']) |
                    pred_type.eq(self.special_types['WEAK'])
                )[0]
                if torch.tensor(sign_here.size()).prod() > 0:
                    if self.channel_to_gloss:
                        ksl_head_input = torch.cat([
                            enc[sign_here],
                            self.channel_tokens[pred_type[sign_here]],
                        ], dim=-1)
                    else:
                        ksl_head_input = enc[sign_here]
                    pred_glosses = self.decode_gloss(
                        self.ksl_head(ksl_head_input),
                        mask=self.sign_mask,
                        dim=-1,
                        keepdims=True,
                        top_k=top_k,
                        top_p=top_p,
                        n_beams=n_beams,
                        return_softmax=True,
                    )
                    pred_ksl[sign_here] = pred_glosses['token']
                    # adjust prediction probability for beam
                    pred_probs[sign_here] = (pred_probs[sign_here] * pred_glosses['probability']).pow(0.5)

                # token 5 (비수지)
                nms_here = torch.where(pred_type.eq(self.special_types['NMS']))[0]
                if torch.tensor(nms_here.size()).prod() > 0:
                    if self.channel_to_gloss:
                        ksl_head_input = torch.cat([
                            enc[nms_here],
                            self.channel_tokens[pred_type[nms_here]],
                        ], dim=-1)
                    else:
                        ksl_head_input = enc[nms_here]
                    nms = self.masked_2d_argmax(self.ksl_head(ksl_head_input), mask=self.nms_mask, dim=-1)['argmax']
                    pred_ksl[nms_here] = nms.unsqueeze(-1)

                # token 6,7 지화
                fs_here = torch.where(
                    pred_type.eq(self.special_types['FS_STRONG']) |
                    pred_type.eq(self.special_types['FS_WEAK'])
                )[0]
                fs_here = fs_here.to(seq.input_ids.device)

                max_len = min(
                    input_ids.shape[0],
                    enc.shape[0],
                    seq.input_ids.shape[0],
                    seq.attention_mask.shape[0]
                )
                valid_mask = fs_here < max_len
                fs_here = fs_here[valid_mask]

                if fs_here.numel() > 0:
                    try:
                        fs = self.point(
                            input_ids[fs_here, :-1],
                            [enc.unsqueeze(1)[fs_here]],
                            argmax=False,
                        )[0]

                        seq_input = seq.input_ids[fs_here]
                        seq_mask = seq.attention_mask[fs_here].to(fs.device)

                        seq_mask[:, 0] = 0
                        pad_here = torch.where(seq_input.eq(self.source_tokenizer.pad_token_id))
                        seq_mask[pad_here] = 0
                        eos_here = torch.where(seq_input.eq(self.source_tokenizer.eos_token_id))
                        seq_mask[eos_here] = 0

                        fs = self.decode_point(
                            p=fs,
                            seq_input=seq_input,
                            seq_mask=seq_mask,
                            dim=1,
                            drop_firstlast=True
                        )

                        for fs_en, fs_i in enumerate(fs_here):
                            pred_fs[fs_i] = fs[fs_en]

                    except RuntimeError as e:
                        print(f"[ERROR] RuntimeError during FS indexing or point decoding: {e}")
                        print(f"input_ids.shape: {input_ids.shape}")
                        print(f"enc.shape: {enc.shape}")
                        print(f"seq.input_ids.shape: {seq.input_ids.shape}")
                        print(f"fs_here: {fs_here}")
                        raise
                    
                    # pred_ksl[fs_here] = pred_type[fs_here].unsqueeze(-1)  # unk = 0

                # token 8 STEP
                step_here = torch.where(pred_type.eq(self.special_types['STEP']))[0]
                if torch.tensor(step_here.size()).prod() > 0:
                    step = torch.tensor([self.ksl_token_series.get('STEP')], device=device)
                    pred_ksl[step_here] = step.unsqueeze(-1)

                pred_duration = self.duration_mod(self.duration_head(enc), 1, 20)

                # parse beam results
                data_index = [i for i, item in enumerate(data) if len(item) >= (beam_i + 1)]
                for pred_i_local, pred_i_global in enumerate(data_index):
                    if pred_type[pred_i_local] in [
                        self.special_types['BOTH'],
                        self.special_types['STRONG'],
                        self.special_types['WEAK'],
                    ]:
                        data_slice[pred_i_local]['prediction'] = []
                        for decode_i in range(n_beams):
                            data_slice[pred_i_local]['prediction'].append({
                                'mode': self.special_types_rev[pred_type[pred_i_local].item()],
                                'gloss': self.dict_token_to_ksl(pred_ksl[pred_i_local, decode_i].item()),
                                'duration': pred_duration[pred_i_local].item(),
                                'probability': pred_probs[pred_i_local, decode_i].item(),
                                'embedding_keys': {
                                    'tokens': pred_ksl[pred_i_local, decode_i].int().unsqueeze(0),
                                    'types': pred_type[pred_i_local].unsqueeze(0).unsqueeze(1),
                                    'durations': pred_duration[pred_i_local],
                                },
                                'past_key_values': self.extract_past_keys(past_key_values, pred_i_local),
                            })
                    else:
                        if pred_type[pred_i_local] in [self.special_types['FS_STRONG'], self.special_types['FS_WEAK']]:
                            data_slice[pred_i_local]['prediction'] = [{
                                'mode': fs_mode_map[pred_type[pred_i_local].item()],
                                'gloss': f'f:{pred_fs[pred_i_local]}',
                                'duration': pred_duration[pred_i_local].item(),
                                'probability': pred_probs[pred_i_local, 0].item(),
                                'embedding_keys': {
                                    'tokens': pred_ksl[pred_i_local, 0].int().unsqueeze(0),
                                    'types': pred_type[pred_i_local].unsqueeze(0).unsqueeze(1),
                                    'durations': pred_duration[pred_i_local],
                                },
                                'past_key_values': self.extract_past_keys(past_key_values, pred_i_local),
                            }]
                        else:
                            data_slice[pred_i_local]['prediction'] = [{
                                'mode': self.special_types_rev[pred_type[pred_i_local].item()],
                                'gloss': self.dict_token_to_ksl(pred_ksl[pred_i_local, 0].item()),
                                'duration': pred_duration[pred_i_local].item(),
                                'probability': pred_probs[pred_i_local, 0].item(),
                                'embedding_keys': {
                                    'tokens': pred_ksl[pred_i_local, 0].int().unsqueeze(0),
                                    'types': pred_type[pred_i_local].unsqueeze(0).unsqueeze(1),
                                    'durations': pred_duration[pred_i_local].unsqueeze(0),
                                },
                                'past_key_values': self.extract_past_keys(past_key_values, pred_i_local),
                            }]

            # re analyze beams
            for row_i in range(len(data)):
                row = data[row_i]
                probabilities = []
                for beam_i, beam in enumerate(row):
                    if len(beam['prediction']) == 0:  # all entries in mb were EOS, so no predictions made
                        probabilities.append((
                            self.beam_probability(beam['probabilities']),
                            beam_i,
                            None,
                        ))
                    else:
                        for pred_i, pred in enumerate(beam['prediction']):
                            probabilities.append((
                                self.beam_probability(beam['probabilities'] + [pred['probability']]),
                                beam_i,
                                pred_i,
                            ))
                probabilities = sorted(probabilities, key=lambda x: x[0], reverse=True)[:n_beams]
                new_row = []
                for prob in probabilities:
                    beam = row[prob[1]]
                    # if eos, just reuse previous for now, optimize later.
                    if len(beam['translation']) > 0 and beam['translation'][-1]['mode'] == 'EOS':
                        beam['prediction'] = []
                        new_row.append(beam)
                    else:
                        pred = beam['prediction'][prob[2]]
                        add_to_translation = {
                            'mode': pred['mode'],
                            'gloss': pred['gloss'],
                            'duration': pred['duration'],
                        }
                        new_row.append({
                            'embedding': self.embed_target(**pred['embedding_keys']),
                            'past_key_values': pred['past_key_values'],
                            'position_ids': beam['position_ids'] + 1,
                            'probabilities': beam['probabilities'] + [pred['probability']],
                            'prediction': [],
                            'translation': beam['translation'] + [add_to_translation],
                            'eos': pred['mode'] == 'EOS',
                        })
                data[row_i] = new_row

            if all([all([beam['translation'][-1]['mode'] == 'EOS' for beam in row]) for row in data]):
                break

            i += 1

        output = [item[0]['translation'][:-1] for item in data]

        if any([len(item) > 0 for item in output]):
            output = [
                self.postprocess(
                    out,
                    source_text=sequences[out_i],
                    nia_data_script=nia_data_script,
                    player_script=player_script,
                ) for out_i, out in enumerate(output)
            ]
            output_dict = defaultdict(list)
            for d in output:
                for k, v in d.items():
                    output_dict[k].append(v)

        else:
            output_dict = {'list': [[] for _ in range(n_seqs)]}

        return output_dict
