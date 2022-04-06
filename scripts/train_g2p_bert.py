import sys
sys.path.insert(0, './')

import os
import argparse
from datetime import datetime
import random
from shutil import copyfile
from collections import defaultdict
import itertools
import statistics

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer

from g2pw.dataset import prepare_data, prepare_pos, TextDataset, get_phoneme_labels, get_char_phoneme_labels
from g2pw.module import G2PW
from g2pw.utils import load_config, RunningAverage, get_logger


def train_batch(model, data, optimizer, device):
    model.train()
    input_ids, token_type_ids, attention_mask, phoneme_mask, char_ids, position_ids, label_ids = \
        [data[name].to(device) for name in ('input_ids', 'token_type_ids', 'attention_mask', 'phoneme_mask', 'char_ids', 'position_ids', 'label_ids')]
    pos_ids = data['pos_ids'].to(device) if model.use_pos else None

    probs, loss, pos_logits = model(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        attention_mask=attention_mask,
        phoneme_mask=phoneme_mask,
        char_ids=char_ids,
        position_ids=position_ids,
        label_ids=label_ids,
        pos_ids=pos_ids
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, valid_loader, device):
    model.eval()

    loss_averager = RunningAverage()
    acc_averager = RunningAverage()
    acc_by_char_averagers = defaultdict(RunningAverage)
    acc_by_char_bopomofo_averagers = defaultdict(RunningAverage)
    pos_acc_averager = RunningAverage()

    with torch.no_grad():
        for data in tqdm(valid_loader, desc='evaluate'):
            input_ids, token_type_ids, attention_mask, phoneme_mask, char_ids, position_ids, label_ids = \
                [data[name].to(device) for name in ('input_ids', 'token_type_ids', 'attention_mask', 'phoneme_mask', 'char_ids', 'position_ids', 'label_ids')]
            pos_ids = data['pos_ids'].to(device) if model.use_pos else None

            probs, loss, pos_logits = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                phoneme_mask=phoneme_mask,
                char_ids=char_ids,
                position_ids=position_ids,
                label_ids=label_ids,
                pos_ids=pos_ids
            )

            loss_averager.add(loss.item())

            corrects = (probs.argmax(dim=-1) == label_ids).cpu().tolist()
            char_ids_cpu = char_ids.cpu().tolist()
            label_ids_cpu = label_ids.cpu().tolist()
            for char_id, label_id, correct in zip(char_ids_cpu, label_ids_cpu, corrects):
                acc_averager.add(correct)
                acc_by_char_averagers[char_id].add(correct)
                acc_by_char_bopomofo_averagers[(char_id, label_id)].add(correct)
            if model.use_pos and pos_logits is not None:
                pos_acc_averager.add_all((pos_logits.argmax(dim=-1) == pos_ids).cpu().tolist())

    acc = acc_averager.get()
    avg_acc_by_char = statistics.mean([averager.get() for averager in acc_by_char_averagers.values()])
    avg_acc_by_char_bopomofo = statistics.mean([averager.get() for averager in acc_by_char_bopomofo_averagers.values()])
    pos_acc = pos_acc_averager.get()
    metric = {
        'avg_loss': loss_averager.get(),
        'acc': acc,
        'avg_acc_by_char': avg_acc_by_char,
        'avg_acc_by_char_bopomofo': avg_acc_by_char_bopomofo,
        'pos_acc': pos_acc
    }
    return metric


def test(config, checkpoint, device, sent_path=None, lb_path=None, pos_path=None):
    if sent_path is None:
        sent_path = config.test_sent_path
    if lb_path is None:
        lb_path = config.test_lb_path
    if pos_path is None:
        pos_path = config.param_pos['test_pos_path']

    tokenizer = BertTokenizer.from_pretrained(config.model_source)

    polyphonic_chars = [line.split('\t') for line in open(config.polyphonic_chars_path).read().strip().split('\n')]
    labels, char2phonemes = get_char_phoneme_labels(polyphonic_chars) if config.use_char_phoneme else get_phoneme_labels(polyphonic_chars)
    chars = sorted(list(char2phonemes.keys()))

    test_texts, test_query_ids, test_phonemes = prepare_data(sent_path, lb_path)
    test_pos_tags = prepare_pos(pos_path) if config.use_pos else None
    test_dataset = TextDataset(tokenizer, labels, char2phonemes, chars, test_texts, test_query_ids, phonemes=test_phonemes, pos_tags=test_pos_tags,
                               use_mask=config.use_mask, use_char_phoneme=config.use_char_phoneme, use_pos=config.use_pos, window_size=config.window_size, for_train=True)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        collate_fn=test_dataset.create_mini_batch,
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

    return evaluate(model, test_loader, device)


def main(config_path):
    config = load_config(config_path, use_default=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Seed and GPU setting
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.manual_seed)
    random.seed(config.manual_seed)
    np.random.seed(config.manual_seed)
    torch.manual_seed(config.manual_seed)

    save_checkpoint_dir = os.path.join('saved_models/', config.exp_name)
    os.makedirs(save_checkpoint_dir, exist_ok=True)

    copyfile(config_path, os.path.join(save_checkpoint_dir, 'config.py'))

    logger_file_path = os.path.join(save_checkpoint_dir, 'record.log')
    logger = get_logger(logger_file_path)

    logger.info(f'device: {device}')
    logger.info(f'now: {datetime.now()}')

    tokenizer = BertTokenizer.from_pretrained(config.model_source)

    polyphonic_chars = [line.split('\t') for line in open(config.polyphonic_chars_path).read().strip().split('\n')]
    labels, char2phonemes = get_char_phoneme_labels(polyphonic_chars) if config.use_char_phoneme else get_phoneme_labels(polyphonic_chars)
    chars = sorted(list(char2phonemes.keys()))

    train_texts, train_query_ids, train_phonemes = prepare_data(config.train_sent_path, config.train_lb_path)
    valid_texts, valid_query_ids, valid_phonemes = prepare_data(config.valid_sent_path, config.valid_lb_path)

    train_pos_tags = prepare_pos(config.param_pos['train_pos_path']) if config.use_pos else None
    valid_pos_tags = prepare_pos(config.param_pos['valid_pos_path']) if config.use_pos else None

    train_dataset = TextDataset(tokenizer, labels, char2phonemes, chars, train_texts, train_query_ids, phonemes=train_phonemes, pos_tags=train_pos_tags,
                                use_mask=config.use_mask, use_char_phoneme=config.use_char_phoneme, use_pos=config.use_pos, window_size=config.window_size, for_train=True)
    valid_dataset = TextDataset(tokenizer, labels, char2phonemes, chars, valid_texts, valid_query_ids, phonemes=valid_phonemes, pos_tags=valid_pos_tags,
                                use_mask=config.use_mask, use_char_phoneme=config.use_char_phoneme, use_pos=config.use_pos, window_size=config.window_size, for_train=True)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        collate_fn=train_dataset.create_mini_batch,
        shuffle=True,
        num_workers=config.num_workers
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        collate_fn=valid_dataset.create_mini_batch,
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
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    i = 1
    is_running = True
    train_loss_averager = RunningAverage()
    best_accuracy = 0.
    checkpoint_path = os.path.join(save_checkpoint_dir, 'best_accuracy.pth')

    while is_running:
        for train_data in train_loader:
            loss = train_batch(model, train_data, optimizer, device)
            train_loss_averager.add(loss)

            if i % config.val_interval == 0:
                # training
                train_loss = train_loss_averager.get()
                train_loss_averager.flush()

                # validation
                metric = evaluate(model, valid_loader, device)

                # save model
                if metric['acc'] > best_accuracy:
                    torch.save(model.state_dict(), checkpoint_path)

                    best_accuracy = metric['acc']

                # log
                pos_acc = 'none' if metric['pos_acc'] is None else metric['pos_acc']
                logger.info(f'[{i}] train_loss={train_loss:.6} valid_loss={metric["avg_loss"]:.6} valid_pos_acc={pos_acc:.6} valid_acc={metric["acc"]:.6} / {metric["avg_acc_by_char"]:.6} / {metric["avg_acc_by_char_bopomofo"]:.6} best_acc={best_accuracy:.6}')
                logger.info(f'now: {datetime.now()}')

            if i >= config.num_iter:
                is_running = False
                break

            i += 1

    if config.test_sent_path and config.test_lb_path:
        logger.info('testing ...')
        logger.info('reloading best accuracy model ...')

        test_metric = test(config, checkpoint_path, device)

        test_pos_acc = 'none' if test_metric['pos_acc'] is None else test_metric['pos_acc']
        logger.info(f'valid_best_acc={best_accuracy:.6} test_loss={test_metric["avg_loss"]:.6} test_pos_acc={test_pos_acc:.6} test_acc={test_metric["acc"]:.6} / {test_metric["avg_acc_by_char"]:.6} / {test_metric["avg_acc_by_char_bopomofo"]:.6}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='config path')
    opt = parser.parse_args()

    main(opt.config)
