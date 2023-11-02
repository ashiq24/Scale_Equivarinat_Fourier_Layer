import random
from torch.utils.data.sampler import Sampler

class CustomRandomSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):

        batch_lists = []
        for cluster_indices in self.data_source.cluster_indices:
            batches = [cluster_indices[i:i + self.batch_size] for i in range(0, len(cluster_indices), self.batch_size)]
            batches = [_ for _ in batches if len(_) == self.batch_size]
            if self.shuffle:
                random.shuffle(batches)
                random.shuffle(cluster_indices)
                
            batch_lists.append(batches)       
        
        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)
        lst = self.flatten_list(lst) 
        return iter(lst)

    def __len__(self):
        return len(self.data_source)