from torch.utils.data import Dataset


class DNADataset(Dataset):
    def __init__(self, X, Y_dev, Y_hk):
        self.X = X
        self.Y_dev = Y_dev
        self.Y_hk = Y_hk

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y_dev[idx], self.Y_hk[idx]
