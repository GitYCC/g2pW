import torch
from transformers import BertConfig

from g2pw.dataset import TextDataset
from g2pw.module import G2PW


class TestG2PW:
    def test_forward(self):
        model_source = 'bert-base-chinese'
        chars = ['這', '是', '為', '了', '測', '試']
        labels = ['ㄘㄜ4', 'ㄕ4', 'ㄩㄥ4']
        model = G2PW.from_pretrained(
            model_source,
            labels=labels,
            chars=chars,
            pos_tags=TextDataset.POS_TAGS,
            use_conditional=False,
            param_conditional=None,
            use_focal=False,
            param_focal=None,
            use_pos=False,
            param_pos=None
        )

        input_ids = torch.tensor(
            [
                [101, 1, 2, 3, 102],
                [101, 4, 5, 6, 102]
            ]
        )
        token_type_ids = torch.tensor(
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]
        )
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1]
            ]
        )
        phoneme_mask = torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 1]
            ]
        )
        char_ids = torch.tensor(
            [0, 1]
        )
        position_ids = torch.tensor(
            [1, 2]
        )
        label_ids = torch.tensor(
            [0, 2]
        )
        logits, loss, pos_logits = model(input_ids, token_type_ids, attention_mask, phoneme_mask, char_ids, position_ids, label_ids=label_ids)
        assert list(logits.size()) == [2, len(labels)]
