import os
from g2pw import G2PWConverter

def ConvertOnnxModel(sentence):
    conv = G2PWConverter(style='pinyin', enable_non_tradional_chinese=True,export_onnx=True)
    print("%s pytorch G2pW:"%sentence,conv(sentence))
    conv = G2PWConverter(style='pinyin', enable_non_tradional_chinese=True,inference_onnx=True)
    print("%s onnx G2pW:"%sentence,conv(sentence))

if __name__ == "__main__":
    # sentence must have polyphonic words
    sentence = "数数"
    ConvertOnnxModel(sentence)
