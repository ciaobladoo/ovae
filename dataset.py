import torch.utils.data as data
import torch


class SmallPatchDataset(data.Dataset):

    def __init__(self, ptf):
        self.data = torch.load(ptf)
        self.batch_size = self.data.size(0)

    def __getitem__(self, idx):
        img = self.data[idx]
        return img

    def __len__(self):
        return self.batch_size