# g2pW: Mandarin Grapheme-to-Phoneme Converter

[![Downloads](https://pepy.tech/badge/g2pw)](https://pepy.tech/project/g2pw)  [![license](https://img.shields.io/badge/license-Apache%202.0-red)](https://github.com/GitYCC/g2pW/blob/master/LICENSE)

**Authors:** [Yi-Chang Chen](https://github.com/GitYCC), Yu-Chuan Chang, Yen-Cheng Chang and Yi-Ren Yeh

This is the official repository of our paper [g2pW: A Conditional Weighted Softmax BERT for Polyphone Disambiguation in Mandarin](https://arxiv.org/abs/2203.10430) (**INTERSPEECH 2022**).

## News

- g2pW is included in [PaddlePaddle](https://github.com/PaddlePaddle)/[PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)
- g2pW is included in [mozillazg](https://github.com/mozillazg)/[pypinyin-g2pW](https://github.com/mozillazg/pypinyin-g2pW)


## Getting Started

### Dependency / Install

(This work was tested with PyTorch 1.7.0, CUDA 10.1, python 3.6 and Ubuntu 16.04.)

- Install [PyTorch](https://pytorch.org/get-started/locally/)

- `$ pip install g2pw`

### Quick Demo

<a href="https://colab.research.google.com/github/GitYCC/g2pW/blob/master/misc/demo.ipynb" target="_blank"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

```python
>>> from g2pw import G2PWConverter
>>> conv = G2PWConverter()
>>> sentence = '上校請技術人員校正FN儀器'
>>> conv(sentence)
[['ㄕㄤ4', 'ㄒㄧㄠ4', 'ㄑㄧㄥ3', 'ㄐㄧ4', 'ㄕㄨ4', 'ㄖㄣ2', 'ㄩㄢ2', 'ㄐㄧㄠ4', 'ㄓㄥ4', None, None, 'ㄧ2', 'ㄑㄧ4']]
>>> sentences = ['銀行', '行動']
>>> conv(sentences)
[['ㄧㄣ2', 'ㄏㄤ2'], ['ㄒㄧㄥ2', 'ㄉㄨㄥ4']]
```

### Load Offline Model

```python
conv = G2PWConverter(model_dir='./G2PWModel-v2-onnx/', model_source='./path-to/bert-base-chinese/')
```

### Support Simplified Chinese and Pinyin

```python
>>> from g2pw import G2PWConverter
>>> conv = G2PWConverter(style='pinyin', enable_non_tradional_chinese=True)
>>> conv('然而，他红了20年以后，他竟退出了大家的视线。')
[['ran2', 'er2', None, 'ta1', 'hong2', 'le5', None, None, 'nian2', 'yi3', 'hou4', None, 'ta1', 'jing4', 'tui4', 'chu1', 'le5', 'da4', 'jia1', 'de5', 'shi4', 'xian4', None]]
```

## Scripts

```
$ git clone https://github.com/GitYCC/g2pW.git
```

### Train Model

For example, we train models on CPP dataset as follows:

```
$ bash cpp_dataset/download.sh
$ python scripts/train_g2p_bert.py --config configs/config_cpp.py
```

### Prediction

```
$ python scripts/test_g2p_bert.py \
    --config saved_models/CPP_BERT_M_DescWS-Sec-cLin-B_POSw01/config.py \
    --checkpoint saved_models/CPP_BERT_M_DescWS-Sec-cLin-B_POSw01/best_accuracy.pth \
    --sent_path cpp_dataset/test.sent \
    --output_path output_pred.txt
```

### Testing

```
$ python scripts/predict_g2p_bert.py \
    --config saved_models/CPP_BERT_M_DescWS-Sec-cLin-B_POSw01/config.py \
    --checkpoint saved_models/CPP_BERT_M_DescWS-Sec-cLin-B_POSw01/best_accuracy.pth \
    --sent_path cpp_dataset/test.sent \
    --lb_path cpp_dataset/test.lb
```

## Checkpoints

- [G2PWModel-v2.zip](https://storage.googleapis.com/esun-ai/g2pW/G2PWModel-v2.zip)
- [G2PWModel-v2-onnx.zip](https://storage.googleapis.com/esun-ai/g2pW/G2PWModel-v2-onnx.zip)

## Citation

To cite the code/data/paper, please use this BibTex
```bibtex
@article{chen2022g2pw,
    author={Yi-Chang Chen and Yu-Chuan Chang and Yen-Cheng Chang and Yi-Ren Yeh},
    title = {g2pW: A Conditional Weighted Softmax BERT for Polyphone Disambiguation in Mandarin},
    journal={Proc. Interspeech 2022},
    url = {https://arxiv.org/abs/2203.10430},
    year = {2022}
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GitYCC/g2pW&type=Date)](https://star-history.com/#GitYCC/g2pW&Date)
