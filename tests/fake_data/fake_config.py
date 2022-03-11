# where to store logs and models
exp_name = 'model1'

# path to lmdb dataset
train_data = ''
valid_data = ''

# for random seed setting
manual_seed = 1313

# number of data loading workers
workers = 4

batch_size = 192

# number of iterations to train for
num_iter = 300000

# interval between each validation
val_interval = 2000

# path to model to continue training
checkpoint = ''

# learning rate, decay rate rho and eps for Adadelta
lr = 1
rho = 0.95
eps = 1e-8

# gradient clipping value
grad_clip = 5

### Data Processing ###
# select training data, e.g. ['data1', 'data2']
select_data = ['data1', 'data2']

# assign ratio for each selected data in the batch, e.g. [0.5, 0.5]
batch_ratio = [0.5, 0.5]

max_seq_len = 100

img_height = 48
img_width = 64

# use rgb input
rgb = False

elements = ''
