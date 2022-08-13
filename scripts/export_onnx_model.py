import sys
sys.path.insert(0, './')

import os
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from g2pw.module import G2PW
from g2pw.dataset import TextDataset, get_phoneme_labels, get_char_phoneme_labels
from g2pw.utils import load_config


def convert_to_onnx(model_dir, save_onnx_path):
    config = load_config(os.path.join(model_dir, 'config.py'), use_default=True)

    polyphonic_chars_path = os.path.join(model_dir, 'POLYPHONIC_CHARS.txt')
    polyphonic_chars = [line.split('\t') for line in open(polyphonic_chars_path).read().strip().split('\n')]
    labels, char2phonemes = get_char_phoneme_labels(polyphonic_chars) if config.use_char_phoneme else get_phoneme_labels(polyphonic_chars)

    chars = sorted(list(char2phonemes.keys()))
    pos_tags = TextDataset.POS_TAGS

    model = G2PW.from_pretrained(
        config.model_source,
        labels=labels,
        chars=chars,
        pos_tags=pos_tags,
        use_conditional=config.use_conditional,
        param_conditional=config.param_conditional,
        use_focal=config.use_focal,
        param_focal=config.param_focal,
        use_pos=config.use_pos,
        param_pos=config.param_pos
    )
    checkpoint = os.path.join(model_dir, 'best_accuracy.pth')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))

    # set the model to inference mode 
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(config.model_source)

    texts = ['重要']
    query_ids = [0]

    dataset = TextDataset(tokenizer, labels, char2phonemes, chars, texts, query_ids,
        use_mask=config.use_mask, use_char_phoneme=config.use_char_phoneme,
        window_size=config.window_size, for_train=False)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        collate_fn=dataset.create_mini_batch,
        num_workers=0
    )
    data = next(iter(dataloader))
    input_names = ['input_ids', 'token_type_ids', 'attention_mask', 'phoneme_mask', 'char_ids', 'position_ids']
    dummy_inputs = tuple([data[name] for name in input_names])

    torch.onnx.export(
        model,
        dummy_inputs,
        save_onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=input_names,
        output_names=['probs'],
        dynamic_axes={'input_ids': [0,1], 'token_type_ids': [0,1], 'attention_mask': [0,1], 
                      'phoneme_mask':[0], 'char_ids': [0], 'position_ids': [0], 'probs': [0]}, 
        verbose=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True, help='model dir path')
    parser.add_argument('--save_onnx_path', required=True, help='path of onnx model')
    opt = parser.parse_args()

    convert_to_onnx(opt.model_dir, opt.save_onnx_path)
 