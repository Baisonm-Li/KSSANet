import torch
import numpy as np
import torch.nn.functional as F


class NPZDataset(torch.utils.data.Dataset):
    def __init__(self,data_path,get_graph=False):
        super(NPZDataset, self).__init__()
        loaded_data = np.load(data_path)
        self.LRHSI_list = loaded_data['LRHSI']
        self.GT_list = loaded_data['GT']

    def __getitem__(self, index,scale=4):
        return torch.from_numpy(self.LRHSI_list[index]).float(), \
                torch.from_numpy(self.GT_list[index]).float()
    
    def __len__(self):
        return len(self.GT_list)

