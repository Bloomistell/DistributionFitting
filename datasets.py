from torch.utils.data import DataLoader, Dataset
import torch



# dataset where each output is drawn from a Gaussian distribution
class GaussianDataset(Dataset):
    def __init__(self):
        super().__init__()

    def generate_data(self, n, n_samples):
        self.n = n
        self.n_samples = n_samples
        self.X = torch.rand(n)
        self.X = self.X.repeat(n_samples).reshape(-1, 1)
        self.y = 0.5*torch.normal(self.X*5, 1) + 0.5*torch.normal(-self.X - 3, 1)

    def __len__(self):
        return self.n * self.n_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]