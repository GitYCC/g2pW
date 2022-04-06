import sys
sys.path.insert(0, './')

import os
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from g2pw.dataset import prepare_data, TextDataset, get_phoneme_labels, get_char_phoneme_labels
from g2pw.module import G2PW
from g2pw.utils import load_config
from train_g2p_bert import test


def main(config, test_sent_path, test_lb_path, test_pos_path, checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_metric = test(config, checkpoint_path, device, sent_path=test_sent_path, lb_path=test_lb_path, pos_path=test_pos_path)

    test_pos_acc = 'none' if test_metric['pos_acc'] is None else test_metric['pos_acc']
    print(f'test_loss={test_metric["avg_loss"]:.6} test_pos_acc={test_pos_acc:.6} test_acc={test_metric["acc"]:.6} / {test_metric["avg_acc_by_char"]:.6} / {test_metric["avg_acc_by_char_bopomofo"]:.6}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='config path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint')
    parser.add_argument('--sent_path', required=False, help='path of *.sent file')
    parser.add_argument('--lb_path', required=False, help='path of *.lb file')
    parser.add_argument('--pos_path', required=False, help='path of *.pos file')

    opt = parser.parse_args()

    config = load_config(opt.config, use_default=True)

    main(config, opt.sent_path, opt.lb_path, opt.pos_path, opt.checkpoint)
