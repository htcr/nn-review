from torch.utils.data.dataset import Dataset
import numpy as np
import scipy.io

class NIST36Dataset(Dataset):
    def __init__(self, partition, data_format='vector'):
        # partition: train, valid, test
        # format: vector, image
        mat_path = '../data/nist36_{}.mat'.format(partition)
        data = scipy.io.loadmat(mat_path)
        x, y = data['{}_data'.format(partition)], data['{}_labels'.format(partition)]
        self.x = x.astype(np.float32) # (examples, features)
        self.y = np.argmax(y, axis=1) # (examples, )
        self.data_format = data_format
        self.mean = np.mean(x)
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        sample = self.x[idx, :] # (features, )
        label = self.y[idx] # scalar

        if self.data_format != 'vector':
            # reshape to img (32, 32)
            sample = sample.reshape(32, 32).transpose()
            sample = sample[np.newaxis, :, :] # chw
        
        sample -= self.mean
        return sample, label