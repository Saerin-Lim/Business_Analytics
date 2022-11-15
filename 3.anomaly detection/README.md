## autoencoder-based anomaly detection 튜토리얼

#### > [autoencoder-based anomaly detection](https://github.com/Saerin-Lim/Business_Analytics/blob/master/3.anomaly%20detection/model-based%20anomaly%20detection%20slide.pdf) 설명 보러가기 <

개인적으로 오토인코더를 사용할 때 가장 애매하게 고민되는 부분은 모델구조 부분이다. 특히 layer를 몇 개나 쌓을건지와 hidden vector의 dimension을 어떻게 설정할지를 많이 고민했었다.

이번 튜토리얼에서는 위와 같은 고민을 해결하기 위해서 layer의 개수와 hidden vector dimension에 따라서 오토인코더의 복원성능과 이상탐지 성능이 어떻게 변화하는지 확인한다.

튜토리얼은 pytorch 라이브러리를 통해 구현한다.

---

### 데이터셋 불러오기

이번 튜토리얼에서 활용할 데이터셋은 대표적인 이미지 데이터인 CIFAR-10을 활용한다. CIFAR-10은 아래 그림처럼 총 10개의 class로 이루어져 있으며 각 class마다 32 by 32 사이즈의 RGB 이미지가 6000개씩 총 60000개의 이미지로 구성되어 있다.

![cifar-10](https://user-images.githubusercontent.com/80674834/201267578-bce70474-3354-4ed1-a081-a39609f134ba.PNG)

데이터는 pytorch를 통해 불러온다.

```py
import numpy as np
import torchvision.datasets as datasets

trainset = datasets.CIFAR10(root='./data', train=True, download=True)
testset = datasets.CIFAR10(root='./data', train=False, download=True)

X = np.concatenate((trainset.data, testset.data), axis=0)
y = np.concatenate((trainset.targets, testset.targets), axis=0)

print(f'X shape : {X.shape} | y shape : {y.shape} | data type : {type(X)}')
```

불러온 데이터의 형태를 보면 X = (60000, 32, 32, 3), y = (60000,)이며 데이터 타입은 ndarray이다.

---

### 이상탐지를 위한 데이터셋 구성

CIFAR-10 데이터는 이상탐지를 위한 데이터가 아닌 이미지 분류를 위한 데이터이다. 따라서 임의로 이상탐지 상황을 가정하고 아래와 같이 데이터 세팅을 진행했다.

- 10개의 class 중 하나를 임의로 선택하여 이상 class로 설정 -> 정상 데이터 : 54000개, 이상 데이터 : 6000개

- 학습/검증/테스트 데이터 분할

  - 학습을 위한 학습 집합(정상 데이터 30000개)

  - threshold 설정을 위한 검증 집합(정상 데이터 12000개 + 이상 데이터 3000개)

  - 이상 탐지 성능 평가를 위한 테스트 집합(정상 데이터 12000개 + 이상 데이터 3000개)

위 세팅을 반영하여 custom dataset을 구성한다. 이 때, 간단한 전처리로 이미지 RGB채널을 평균 0.5, 표준편차 0.5로 nomalize한다.

```py
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Custom_dataset(Dataset):
    def __init__(self, X, y, mode='train'):
        super().__init__()
        
        # define transform for preprocessing
        self.transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        # randomly choose abnormal class & get instance indices
        self.abnormal_class = random.choice(np.unique(y))
        
        normal_idx = np.argwhere(y!=self.abnormal_class).reshape(-1)
        abnormal_idx = np.argwhere(y==self.abnormal_class).reshape(-1)
        
        # split train/valid/test using indices
        np.random.shuffle(normal_idx)
        np.random.shuffle(abnormal_idx)
        train_idx = normal_idx[:30000] # of normal :30000
        valid_idx = np.concatenate(normal_idx[30000:42000], # of normal   : 12000
                                   abnormal_idx[:3000])     # of abnormal : 3000
        
        test_idx = np.concatenate(normal_idx[42000:],       # of normal   : 12000
                                   abnormal_idx[3000:])     # of abnormal : 3000
        
        # define dataset according to mode
        if mode == 'train':
            self.X, self.y = X[train_idx], y[train_idx]
        elif mode == 'valid':
            self.X, self.y = X[valid_idx], y[valid_idx]
        else:
            self.X, self.y = X[test_idx], y[test_idx]
    
    def __getitem__(self, idx):
        return {'X':self.transform(self.X[idx]), 'y':self.y[idx]}
    
    def __len__(self):
        return len(self.y)
        
```

---

### 오토인코더 

다음으로 오토인코더 모델 코드를 작성한다. 오토인코더의 각 layer는 아래 그림처럼 가장 기본적인 linear layer를 활용한다.

이 때, layer수와 hidden dimension을 자유롭게 조절 가능하도록 만든다.

![autoencoder](https://user-images.githubusercontent.com/80674834/201868581-fb55bd31-562a-40c2-b0ce-32e396b2cca5.PNG)

먼저 인코더 코드를 작성한다. 인코더는 최종 hidden dimension과 각 layer마다의 dimension을 list형태인 layer_dims로 입력받는다.

make_layers 함수를 통해 입력 이미지에서 각 layer_dims에 있는 dimension으로 매핑하는 linear layer가 생성된다. 이 때, 활성화 함수는 ReLU를 사용한다.

그리고 최종적으로 hidden dimension으로 매핑해주는 마지막 linear layer를 생성하면 총 len(layer_dims)+1개의 linear layer로 구성된 인코더를 얻을 수 있다.

```py
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, hid_dim=256, layer_dims:list=[2048,1024,512]):
        super().__init__()
                
        self.enc = self.make_layers(layer_dims)
        self.fc = nn.Linear(layer_dims[-1], hid_dim)
    
    def forward(self, x):
        """x : batch, C, W, H"""
        x = self.enc(x) # batch, layer_dims[-1]
        x = self.fc(x) # batch, hid dim
        return x
    
    def make_layers(self, layer_dims):
        layers = []
        input_dim = 3*32*32
        for dim in layer_dims:
            layers += [nn.Linear(input_dim, dim), nn.ReLU(True)]
            input_dim = dim
        return nn.Sequential(*layers)
```

디코더는 정확히 인코더와 대칭을 이루도록 작성한다. hid_dim에서 시작해서 layer_dims의 역순으로 점점 커지는 구조를 가진다.

make_layers 함수에서 layer_dims를 역순으로 바꾸고 encoder와 같이 linear layer와 ReLU 함수 시퀀스를 생성한다.

마지막으로 이미지 데이터 공간인 3*32*32차원으로 이미지를 매핑해주는 linear layer를 추가한 뒤, 3by32by32 형태로 reshape을 진행한다.

```py
class Decoder(nn.Module):
    def __init__(self, hid_dim=256, layer_dims:list=[2048,1024,512]):
        super().__init__()
                
        self.enc = self.make_layers(hid_dim, layer_dims)
        self.fc = nn.Linear(layer_dims[0], 3*32*32)
    
    def forward(self, x):
        """x : batch, hid dim"""
        x = self.enc(x) # batch, layer_dims[0]
        x = self.fc(x) # batch, 3*32*32
        
        # reshape 1d vector to image
        x = x.reshape(-1,3,32,32)
        
        return x
    
    def make_layers(self, start_dim, layer_dims):
        layers = []
        layer_dims = layer_dims[::-1]
        input_dim = start_dim
        for dim in layer_dims:
            layers += [nn.Linear(input_dim, dim), nn.ReLU(True)]
            input_dim = dim
        return nn.Sequential(*layers)
```

앞에서 작성한 인코더를 하나로 합쳐서 최종적인 오토인코더를 생성한다. 이 때, 원활한 학습을 위해서 he initialization으로 가중치를 초기화한다.

```py
from utils import initialize_weights

class Autoencoder(nn.Module):
    def __init__(self, hid_dim=256, layer_dims:list=[2048,1024,512]):
        super().__init__()
                
        self.enc = Encoder(hid_dim=hid_dim, layer_dims=layer_dims)
        self.dec = Decoder(hid_dim=hid_dim, layer_dims=layer_dims)
        
        self.apply(initialize_weights)
    
    def forward(self, x):
        """x: B,3,32,32"""
        x = self.enc(x) # B, hid dim
        x = self.dec(x) # B, 3, 32, 32
        return x
```

---

