root = './cpp_dataset/'

manual_seed = 1313
model_source = 'bert-base-chinese'
polyphonic_chars_path = root + 'POLYPHONIC_CHARS.txt'
window_size = 32
num_workers = 2
use_mask = True
use_conditional = True
param_conditional = {
    'bias': True,
    'char-linear': True,
    'pos-linear': False,
    'char+pos-second': True,
}

# for training
exp_name = 'CPP_BERT_M_DescWS-Sec-cLin-B_POSw01'
train_sent_path = root + 'train.sent'
train_lb_path = root + 'train.lb'
valid_sent_path = root + 'dev.sent'
valid_lb_path = root + 'dev.lb'
test_sent_path = root + 'test.sent'
test_lb_path = root + 'test.lb'
batch_size = 256
lr = 5e-5
val_interval = 200
num_iter = 10000
use_pos = True
param_pos = {
    'weight': 0.1,
    'pos_joint_training': True,
    'train_pos_path': root + 'train.pos',
    'valid_pos_path': root + 'dev.pos',
    'test_pos_path': root + 'test.pos'
}
