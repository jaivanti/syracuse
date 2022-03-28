import random

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision.transforms import Compose, ToTensor


class OnlineDataset(Dataset):
    def __init__(self, max_size=2048):
        self.max_size = max_size  #

        self.data = []
        self.transform = Compose([ToTensor(), ]) #tranforming into tensor

    def __len__(self):
        return len(self.data) #data lenght

    def __getitem__(self, index) -> T_co: #giving index of the dataset
        index = random.randrange(0, len(self.data))
        return self.data[index]

    def empty(self) -> bool:
        return len(self) == 0

    def size(self) -> int:
        return len(self)

    def put(self, lr, hr):
        if len(self) >= self.max_size:
            self.data = self.data[self.max_size // 4:]
        lr = self.transform(lr)
        hr = self.transform(hr)
        self.data.append((lr, hr))
