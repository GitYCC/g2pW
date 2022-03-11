from unittest.mock import Mock

import torch

from g2pw.dataset import prepare_data, TextDataset


def test_prepare_data():
    sent_path = 'tests/fake_data/fake.sent'
    lb_path = 'tests/fake_data/fake.lb'
    texts, query_ids, phonemes = prepare_data(sent_path, lb_path)
    assert texts == [
        '华盛顿政府对威士忌酒暴乱的镇压得到了广泛的认同。',
        '类似的化合物还有乙酰乙酸乙酯、丙二酸二甲酯、乙酸乙酯、乙酰丙酮等。',
        '”、“告子为仁，譬犹跂以为长，隐以为广，不可久也”等等。'
    ]
    assert query_ids == [17, 12, 10]
    assert phonemes == ['le5', 'yi3', 'qi3']


class TestTextDataset:
    def test_getitem(self):
        def convert_tokens_to_ids(tokens):
            mapping = {
                '[PAD]': 0,
                '[CLS]': 101,
                '[SEP]': 102,
                '得': 1,
                '到': 2,
                '了': 3,
                '认': 4,
                '同': 5,
                '乙': 6,
                '酸': 7,
                '酯': 8
            }
            return [mapping[token] for token in tokens]
        mocked_tokenizer = Mock()
        mocked_tokenizer.tokenize = lambda word: [word]
        mocked_tokenizer.convert_tokens_to_ids = convert_tokens_to_ids

        labels = ['le5', 'yi3', 'qi3']
        texts = [
            '得到了认同',
            '乙酸乙酯',
        ]
        query_ids = [2, 2]
        phonemes = ['le5', 'yi3']
        char2phonemes = None
        chars = ['了', '乙']

        dataset = TextDataset(mocked_tokenizer, labels, char2phonemes, chars, texts, query_ids, phonemes=phonemes,
                              use_mask=False, use_char_phoneme=False, window_size=10, for_train=True)
        data = dataset[0]

        expected_input_ids = torch.tensor([101, 1, 2, 3, 4, 5, 102])
        expected_position_id = 2 + 1  # because of [CLS]
        expected_label_id = 0

        assert torch.all(torch.eq(data['input_ids'], expected_input_ids))
        assert data['position_id'] == expected_position_id
        assert data['label_id'] == expected_label_id
