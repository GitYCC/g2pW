import sys
sys.path.insert(0, './')

import os
import argparse
import math

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

from g2pw.dataset import prepare_data, TextDataset, get_phoneme_labels, get_char_phoneme_labels, ANCHOR_CHAR
from g2pw.module import G2PW
from g2pw.utils import load_config
from g2pw.api import predict


def main(config, checkpoint, sent_path, output_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(config.model_source)

    polyphonic_chars = [line.split('\t') for line in open(config.polyphonic_chars_path).read().strip().split('\n')]
    labels, char2phonemes = get_char_phoneme_labels(polyphonic_chars) if config.use_char_phoneme else get_phoneme_labels(polyphonic_chars)

    chars = sorted(list(char2phonemes.keys()))

    texts, query_ids = prepare_data(sent_path)

    dataset = TextDataset(tokenizer, labels, char2phonemes, chars, texts, query_ids,
                          use_mask=config.use_mask, use_char_phoneme=config.use_char_phoneme, window_size=config.window_size, for_train=False)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        collate_fn=dataset.create_mini_batch,
        num_workers=config.num_workers
    )

    model = G2PW.from_pretrained(
        config.model_source,
        labels=labels,
        chars=chars,
        pos_tags=TextDataset.POS_TAGS,
        use_conditional=config.use_conditional,
        param_conditional=config.param_conditional,
        use_focal=config.use_focal,
        param_focal=config.param_focal,
        use_pos=config.use_pos,
        param_pos=config.param_pos
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    preds, confidences = predict(model, dataloader, device, labels)
    if config.use_char_phoneme:
        preds = [pred.split(' ')[1] for pred in preds]

    for text, query_id, pred, confidence in zip(texts, query_ids, preds, confidences):
        print('{font}{anchor}{mid}{anchor}{end},{pred},{confidence:.5}'.format(
            anchor=ANCHOR_CHAR,
            font=text[:query_id],
            mid=text[query_id],
            end=text[query_id+1:],
            pred=pred,
            confidence=confidence
        ))
    if output_path:
        lines = [f'{pred},{confidence}' for pred, confidence in zip(preds, confidences)]
        open(output_path, 'w').write('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='config path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint')
    parser.add_argument('--sent_path', required=True, help='path of *.sent file')
    parser.add_argument('--output_path', required=False, help='path of prediction results')
    opt = parser.parse_args()

    config = load_config(opt.config, use_default=True)

    main(config, opt.checkpoint, opt.sent_path, output_path=opt.output_path)
