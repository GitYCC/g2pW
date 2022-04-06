from ckiptagger import WS, POS
# import opencc
from tqdm import tqdm
import numpy as np

CKIP_DATA = './data/'
ANCHOR_CHAR = '▁'

ws = WS(CKIP_DATA)
pos = POS(CKIP_DATA)

# converter = opencc.OpenCC('s2t.json')

root = './dataset/'
path = f'{root}/dev.sent'
out_path = f'{root}/dev.pos'

lines = open(path).read().strip().split('\n')
sentence_list = [line.replace(ANCHOR_CHAR, '') for line in lines]
position_list = [line.index(ANCHOR_CHAR) for line in lines]

mapping = {
    'A': 'A',
    'Caa': 'C', 'Cab': 'C', 'Cba': 'C', 'Cbb': 'C',
    'D': 'D', 'Da': 'D', 'Dfa': 'D', 'Dfb': 'D', 'Di': 'D', 'Dk': 'D', 'DM': 'D',
    'I': 'I',
    'Na': 'N', 'Nb': 'N', 'Nc': 'N', 'Ncd': 'N', 'Nd': 'N', 'Nep': 'N', 'Neqa': 'N', 'Neqb': 'N', 'Nes': 'N', 'Neu': 'N', 'Nf': 'N', 'Ng': 'N', 'Nh': 'N', 'Nv': 'N',
    'P': 'P',
    'T': 'T',
    'VA': 'V', 'VAC': 'V', 'VB': 'V', 'VC': 'V', 'VCL': 'V', 'VD': 'V', 'VF': 'V', 'VE': 'V', 'VG': 'V', 'VH': 'V', 'VHC': 'V', 'VI': 'V', 'VJ': 'V', 'VK': 'V', 'VL': 'V', 'V_2': 'V',
    'DE': 'DE',
    'SHI': 'SHI',
}
# map to: ['A', 'C', 'D', 'I', 'N', 'P', 'T', 'V', 'DE', 'SHI']


batch_size = 2048

fw = open(out_path, 'w', buffering=batch_size)

i = 0

for _ in tqdm(list(range(int(np.ceil(len(lines)/batch_size))))):
    j = min(len(lines), i+batch_size)
    sentences = [sent for sent in sentence_list[i:j]]
    positions = position_list[i:j]

    word_sentence_list = ws(sentences)
    pos_sentence_list = pos(word_sentence_list)

    annotations = []
    for position, sentence, words, tags in zip(positions, sentences, word_sentence_list, pos_sentence_list):
        annotation = None
        x = -1
        for word, tag in zip(words, tags):
            for _ in range(len(word)):
                x += 1
                if x == position:
                    annotation = mapping.get(tag, 'UNK')
                    break
            if annotation:
                break
        assert annotation is not None
        annotations.append(annotation)

    if i > 0:
        fw.write('\n')
    fw.write('\n'.join(annotations))
    i = j

fw.close()
