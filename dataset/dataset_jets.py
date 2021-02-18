from dataset import *
from torch.utils.data import *

#from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import os
from matplotlib import pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq


# def read_data_sets()
#   class DataSets(object):
#     pass

#   data_sets = DataSets()

    
    
# def get_X(data):
#     X = data['X_jets'].cuda()
#     return X


class ParquetDataset(Dataset):
    def __init__(self, filename):
        self.parquet = pq.ParquetFile(filename)
        self.cols = None 
    def __getitem__(self, index):
        data = self.parquet.read_row_group(index, columns=self.cols).to_pydict()
        data['X_jet'] = np.float32(data['X_jet'][0]) 
        # Preprocessing
        data['X_jet'] = data['X_jet'][:, 20:105, 20:105]
        data['X_jet'][data['X_jet'] < 1.e-3] = 0. # Zero-Suppression
        data['X_jet'] = data['X_jet'][0][1]
        
        return dict(data)
    def __len__(self):
        return self.parquet.num_row_groups


class JetsDataset(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        data_file = r'C:\Users\Darya\cernbox_pq_imgs'
        self.datasets = [os.path.join(data_file, file) for file in os.listdir(data_file) if file.split('.')[-1] == 'parquet']        
#         self.jets = input_data.read_data_sets(data_file, one_hot=True)
        self.data_dims = [84, 84, 1]
        self.name = "jets"
        self.batch_size = 200
        self.range = [0.0, 1000.0]
        
        dset = ConcatDataset([ParquetDataset(dataset) for dataset in self.datasets])
        self.train_cut = int(0.8 * len(dset))
        idxs = np.random.permutation(len(dset))
        data_sampler = sampler.SubsetRandomSampler(idxs[:self.train_cut])
        self.data_loader = DataLoader(dataset=dset, batch_size=self.batch_size, shuffle=False, num_workers=0, sampler=data_sampler, pin_memory=True)
    
    
    def next_batch(self, batch_size):
        return next(iter(self.data_loader))

    def next_test_batch(self, batch_size):
        return next(iter(self.data_loader))
        
    def display(self, image):
        return np.clip(image) 
                    
#     def reset(self):
#         self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


        
# if __name__ == '__main__':
#     jets_data = JetsDataset()
    
#     train_loader = JetsDataset.train_test_loader()
#     for i, data in enumerate(train_loader):
#         sample = get_X(data)
#     #while True:
#     #    sample = jets_data.next_batch(100)
#         for index in range(9):
#             plt.subplot(3, 3, index + 1)
#             plt.imshow(sample[index, :, :, 0].astype(np.float), cmap=plt.get_cmap('Greys'))
#         plt.show()
