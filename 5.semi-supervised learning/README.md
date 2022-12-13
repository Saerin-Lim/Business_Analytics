
## Self-training 튜토리얼

Self-training은 간단한 semi-supervised learning 방법론 중 하나로 pseudo labeling을 이용한 방법이다.

Self-training은 unlabeled data에 대한 추론을 한 뒤, 특정 클래스에 속할 확률이 threshold보다 높으면 pseudo label로 활용한다.

전체적인 알고리즘을 아래와 같다.

1. labeled data로 모델을 학습

2. 학습된 모델을 통해서 unlabeled data 추론

3. 추론 확률이 threshold를 넘는 unalbeled data에 대해 pseudo labeling 진행

4. labeled data와 pseudo-labeled data를 통해서 모델 재학습

5. unlabeled data에 모두 pseudo-label이 부여되거나 종료 조건을 만족하면 알고리즘 종료

![image](https://user-images.githubusercontent.com/80674834/207237944-07a86aae-d30a-4b90-9ac0-1a834ec0f61c.png)

이 과정에서 pseudo labeling을 할 unlabeled data를 정하는 전략은 중요한 hyperparameter 중 하나이다.

일반적으로 확률값이 특정 threshold 이상 (ex.0.9 이상)인 unlabeled data에 대해 pseudo labeling을 하거나, 확률값이 높은 top-k unlabeled data를 뽑는 방법이 있으며, 두가지를 동시에 사용하는 방법도 있다.

본 튜토리얼에서는 self-training을 from scratch로 코드를 작성하고, threshold 전략, top-k 전략, 그리고 둘 모두를 활용한 threshold + top-k 전략 중 어떤 전략이 가장 성능이 좋은지 확인한다.

최종적으로 전략이 선택되었다면 labeled data 개수에 따라서 self-training 성능이 어떻게 변하는지 확인하고 supervised learning과 비교한다.

---

### 활용 데이터

이번 튜토리얼에서 활용할 데이터셋은 CIFAR-10으로 아래의 그림처럼 총 10개의 클래스로 구성된 이미지 데이터셋이다. 이미지의 크기는 3by32by32 이며, 각 클래스별로 6000장 씩 총 60000장의 이미지가 존재한다. 

![image](https://user-images.githubusercontent.com/80674834/207241859-0a1c56b6-33fa-4fce-8d60-79e012b66899.png)

---

### hyperparameters 세팅(Utils -> config.py)

본격적인 코드 작성에 앞서서 실험에 필요한 hyperparameter를 정의하고 Config class를 통해서 접근이 용이하도록 한다.
```py
class Config(object):
    def __init__(self) -> None:
        
        # experiment hyperparameters
        self.seed = 2022
        
        # data hyperparameters
        self.num_l = 25000
        self.num_u = 50000 - self.num_l
        
        # train hyperparameters
        self.batch_size = 64 
    
```

---

### Custom Dataset 구축 (datasets.py)

CIFAR-10은 pytorch에 내장되어 있어 쉽게 dataset을 구축할 수 있다. 하지만 self-training을 위해서는 unlabeled data의 index가 있어야지만 unlabeled data에서 pseudo label을 할 data를 식별 할 수 있다. 

따라서 아래와 같이 custom Dataset을 작성했다. transform을 통해 이미지를 nomalize하였고, getitem에서 이미지의 index까지 반환하도록 작성하였다.

```py
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class CIFAR10_Dataset(Dataset):
    def __init__(self, mode:str='train'):
        super().__init__()
        
        self.transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
        
        if mode == 'train':
            dataset = datasets.CIFAR10(root='./Data', train=True, download=False)
        else:
            dataset = datasets.CIFAR10(root='./Data', train=False, download=False)
            
        self.X, self.y = dataset.data, dataset.targets
    
    def __getitem__(self, index):
        X, y = self.transform(self.X[index]), self.y[index]
        return dict(idx=index, X=X, y=y)
    
    def __len__(self):
        return len(self.y)
```

### Inital Dataloader & Update Dataloader(dataloaders.py)

CIFAR-10은 기본적으로 학습 데이터와 테스트 데이터가 50000장/10000장으로 분할되어 있다. 여기에 semi-supervised scenario를 만들기 위해서 학습 데이터 중 특정 개수만큼 labeled dataset과 unlabeled dataset을 만들어야 한다.

또한, self-training 과정에서 생성되는 pseudo label과 해당하는 이미지는 labeled dataset으로 옮겨짐과 동시에 unlabeled dataset에서 삭제되어야 한다.

아쉽게도 pytorch에서는 동적인 sampling을 지원하지 않기 때문에 학습 중간중간(pseudo label을 생성할 때) labeled dataset과 unlabeled dataset을 재정의하고, dataloader를 새로 업데이트 해줘야 한다.

본 튜토리얼에서는 SubsetRandomSampler라는 함수를 이용하여 이 과정을 구현한다. SubsetRandomSampler는 가져오길 원하는 부분집합 데이터의 index만 입력하면 부분집합만 데이터를 가져올 수 있도록 하는 sampling 함수이다.

이 과정에서 init_dataloaders 함수와 update_dataloaders 함수를 작성하였다.

먼저 init_dataloaders 함수는 학습을 시작하기 전, dataloader를 초기화 한다. 즉, 전체 학습 데이터에서 labeled dataset과 unlabeled dataset을 특정 개수만큼(args.num_l) 분할한다.

다음으로 update_dataloaders 함수는 pseudo label이 생성되면 해당하는 데이터를 labeled dataset에 추가함과 동시에 unlabeled dataset에서 제거하고, 새로운 label data loader와 unlabeled data loader를 정의한다.

```py
import random
from torch.utils.data import DataLoader, SubsetRandomSampler

from datasets import CIFAR10_Dataset

def init_dataloaders(args):
    
    # load datasets
    trainset = CIFAR10_Dataset(mode='train')
    testset = CIFAR10_Dataset(mode='test')
    
    # define label & unlabel indicies
    total_indices = list(range(len(trainset.targets)))
    label_indices = random.sample(total_indices, args.num_l)
    unlabel_indices = list(set(total_indices)-set(label_indices))
    
    # make sampler
    label_sampler = SubsetRandomSampler(label_indices)
    unlabel_sampler = SubsetRandomSampler(unlabel_indices)
    
    print(f'# of train data : {len(total_indices)} | # of test data : {len(testset.targets)}')
    print(f'# of labeled data in trainset : {len(label_indices)} | # of unlabeled data in trainset : {len(unlabel_indices)}')
    
    # make dataloader
    label_loader = DataLoader(trainset, batch_size=args.batch_size, sampler=label_sampler)
    unlabel_loader = DataLoader(trainset, batch_size=args.batch_size, sampler=unlabel_sampler)
    test_loader = DataLoader(testset, batch_size=args.batch_size)
    
    loaders = dict(labeled=label_loader,
                   unlabeled=unlabel_loader,
                   test=test_loader,
                   num_pseudo=[])
    
    indices = dict(total=total_indices,
                   labeled=label_indices,
                   unlabeled=unlabel_indices)
    
    return loaders, indices, trainset

def update_dataloaders(args, loaders, indices, pseudo_indices, trainset):
    
    # update label & unlabel indices
    indices['labeled'] = indices['labeled'] + pseudo_indices
    indices['unlabeled'] = list(set(indices['total'])-set(indices['labeled']))
    indices['num_pseudo'].append(len(pseudo_indices))

    # update sampler
    label_sampler = SubsetRandomSampler(indices['labeled'])
    unlabel_sampler = SubsetRandomSampler(indices['unlabeled'])

    # make new dataloader
    loaders['labeled'] = DataLoader(trainset, batch_size=args.batch_size, sampler=label_sampler)
    loaders['unlabeled'] = DataLoader(trainset, batch_size=args.batch_size, sampler=unlabel_sampler)
    
    return loaders, indices
```

---

### Resnet18

다음으로 학습 모델을 구축하였다. 이미지 데이터이기 때문에 CNN기반 모델 중 선택했으며, 그 중 가장 많이 활용되는 Resnet18을 선택했다.

해당 코드는 아래와 같으며, [해당 github](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)에서 가지고 왔다.

```py
import torch.nn as nn

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
```


