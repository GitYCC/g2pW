from unittest.mock import Mock

import pytest

from g2pw.utils import load_config, tokenize_and_map, wordize_and_map


@pytest.mark.parametrize(
    'text, expected_words, expected_text2word, expected_word2text',
    [
        (
            '我 愛  喝  apple juice ',
            ['我', '愛', '喝', 'apple', 'juice'],
            [0, None, 1, None, None, 2, None, None, 3, 3, 3, 3, 3, None, 4, 4, 4, 4, 4, None],
            [(0, 1), (2, 3), (5, 6), (8, 13), (14, 19)]
        ),
        (
            'Rolling in the deep ! !',
            ['Rolling', 'in', 'the', 'deep', '!', '!'],
            [0, 0, 0, 0, 0, 0, 0, None, 1, 1, None, 2, 2, 2, None, 3, 3, 3, 3, None, 4, None, 5],
            [(0, 7), (8, 10), (11, 14), (15, 19), (20, 21), (22, 23)]
        ),
        (
            'I am YC.',
            ['I', 'am', 'YC', '.'],
            [0, None, 1, 1, None, 2, 2, 3],
            [(0, 1), (2, 4), (5, 7), (7, 8)]
        ),
        (
            '未知：? unknown: ?',
            ['未', '知', '：', '?', 'unknown', ':', '?'],
            [0, 1, 2, 3, None, 4, 4, 4, 4, 4, 4, 4, 5, None, 6],
            [(0, 1), (1, 2), (2, 3), (3, 4), (5, 12), (12, 13), (14, 15)]
        )
    ]
)
def test_wordize_and_map(text, expected_words, expected_text2word, expected_word2text):
    words, text2word, word2text = wordize_and_map(text)
    assert words == expected_words
    assert text2word == expected_text2word
    assert word2text == expected_word2text


@pytest.mark.parametrize(
    'sentence, expected_tokens, expected_text2token, expected_token2text',
    [
        (
            '我 愛  喝  apple juice ',
            ['我', '愛', '喝', 'apple', 'j', '##ui', '##ce'],
            [0, None, 1, None, None, 2, None, None, 3, 3, 3, 3, 3, None, 4, 5, 5, 6, 6, None],
            [(0, 1), (2, 3), (5, 6), (8, 13), (14, 15), (15, 17), (17, 19)]
        ),
        (
            'Rolling in the deep ! !',
            ['rolling', 'in', 'the', 'deep', '!', '!'],
            [0, 0, 0, 0, 0, 0, 0, None, 1, 1, None, 2, 2, 2, None, 3, 3, 3, 3, None, 4, None, 5],
            [(0, 7), (8, 10), (11, 14), (15, 19), (20, 21), (22, 23)]
        ),
        (
            'I am YC.',
            ['i', 'am', 'y', '##c', '.'],
            [0, None, 1, 1, None, 2, 3, 4],
            [(0, 1), (2, 4), (5, 6), (6, 7), (7, 8)]
        ),
        (
            '未知：? unknown: ?',
            ['未', '知', '：', '[UNK]', 'u', '##nk', '##now', '##n', ':', '[UNK]'],
            [0, 1, 2, 3, None, 4, 5, 5, 6, 6, 6, 7, 8, None, 9],
            [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 8), (8, 11), (11, 12), (12, 13), (14, 15)]
        )
    ]
)
def test_tokenize_and_map(sentence, expected_tokens, expected_text2token, expected_token2text):
    def tokenize(word):
        mapping = {
            'juice': ['j', '##ui', '##ce'],
            'YC': ['y', '##c'],
            '?': ['[UNK]'],
            'unknown': ['u', '##nk', '##now', '##n']
        }
        if word in mapping:
            return mapping[word]
        else:
            return [word.lower()]

    mocked_tokenizer = Mock()
    mocked_tokenizer.tokenize = tokenize

    tokens, text2token, token2text = tokenize_and_map(mocked_tokenizer, sentence)

    assert tokens == expected_tokens
    assert text2token == expected_text2token
    assert token2text == expected_token2text


def test_load_config():
    path = 'tests/fake_data/fake_config.py'

    config = load_config(path)

    assert config.exp_name == 'model1'
    assert config.batch_size == 192
    assert config.rho == 0.95
    assert config.select_data == ['data1', 'data2']
    assert config.rgb == False
