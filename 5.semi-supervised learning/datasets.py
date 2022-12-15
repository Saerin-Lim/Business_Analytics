from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CIFAR10_Dataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                 (0.2023, 0.1994, 0.2010))])

        self.X, self.y = X, y
    
    def __getitem__(self, index):
        X, y = self.transform(self.X[index]), self.y[index]
        return dict(idx=index, X=X, y=y)
    
    def __len__(self):
        return len(self.y)

            
            