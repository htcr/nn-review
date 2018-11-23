import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr
import os

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x_raw = train_data['train_data']
valid_x_raw = valid_data['valid_data']

# normalize
train_x_raw -= np.mean(train_x_raw, axis=1, keepdims=True)
valid_x_raw -= np.mean(valid_x_raw, axis=1, keepdims=True)

dim = 32
# do PCA

train_x = train_x_raw.transpose() # (num_features, num_examples makes more sense)
covariance = train_x @ train_x.transpose() # (num_features, num_features)
U, S, Vt = np.linalg.svd(covariance) # now the columns of U are eigenvectors of covariance, the most significant one on the left
projection = U[:, :dim] # (num_features, keep_dim) matrix used for compressing and reconstructiong data

# rebuild a low-rank version
lrank = projection.transpose() @ train_x

# rebuild it
recon = projection @ lrank
recon = recon.transpose() # restore to hw format

class_ids = [0, 1, 2, 3, 4]
item_ids = [[2, 6], [0, 5], [1, 2], [3, 4], [1, 6]]

output_dir = 'PCA_samples'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# build valid dataset
recon_valid = projection @ projection.transpose() @ valid_x_raw.transpose()
recon_valid = recon_valid.transpose()

for c in class_ids:
    for i in item_ids[c]:
        plt.subplot(2,1,1)
        ori_img = valid_x_raw[c*100+i].reshape(32,32).T
        plt.imshow(ori_img)
        plt.subplot(2,1,2)
        rec_vis_img = recon_valid[c*100+i].reshape(32,32).T
        # ensures that visualization between reconstruced
        # and original images will not be too different because
        # of difference in min/max values
        rec_vis_img = np.clip(rec_vis_img, np.min(ori_img), np.max(ori_img))
        plt.imshow(rec_vis_img)
        # plt.show()
        plt.savefig(os.path.join(output_dir, 'class_{}_{}.png'.format(c, i)))

total = []
for pred,gt in zip(recon_valid,valid_x_raw):
    total.append(psnr(gt,pred))
print(np.array(total).mean())