from torch.utils.data.dataset import Dataset
import numpy as np
import scipy.io

class NIST36Dataset(Dataset):
    def __init__(self, partition, format='vector'):
        # partition: train, valid, test
        # format: vector, image
        mat_path = '../data/nist36_{}.mat'.format(partition)
        data = scipy.io.loadmat(mat_path)
        x, y = data['{}_data'.format(partition)], data['{}_labels'.format(partition)]
        self.x = x # (examples, features)
        self.y = np.argmax(y, axis=1) # (examples, )
        self.format = format
        self.mean = np.mean(x)
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        sample = self.x[idx, :] # (features, )
        label = self.y[idx] # scalar

        if self.format != 'vector':
            # reshape to img (32, 32)
            sample = sample.reshape(32, 32).transpose()
            sample = sample[np.newaxis, :, :] # chw
        
        sample -= self.mean
        return sample, label