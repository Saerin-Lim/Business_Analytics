## autoencoder-based anomaly detection 튜토리얼

#### > [autoencoder-based anomaly detection](https://github.com/Saerin-Lim/Business_Analytics/blob/master/3.anomaly%20detection/model-based%20anomaly%20detection%20slide.pdf) 설명 보러가기 <

이번 튜토리얼에서는 이미지 데이터에 대해서 오토인코더 기반 이상 탐지 프레임워크를 from scratch로 작성하는 것을 목표로 한다.

특히 지금까지 찾아본 오토인코더 기반 이상 탐지 튜토리얼 중 검증 집합을 통한 grid search를 통해 threshold를 결정하는 튜토리얼은 없었다.

따라서 threshold grid search를 추가한 오토인코더 기반 이상 탐지 프레임워크 튜토리얼을 진행한다.

---

### 데이터셋 불러오기

이번 튜토리얼에서 활용할 데이터셋은 대표적인 이미지 데이터인 CIFAR-10을 활용한다. CIFAR-10은 아래 그림처럼 총 10개의 class로 이루어져 있으며 각 class마다 32 by 32 사이즈의 RGB 이미지가 6000개씩 총 60000개의 이미지로 구성되어 있다.

![cifar-10](https://user-images.githubusercontent.com/80674834/201267578-bce70474-3354-4ed1-a081-a39609f134ba.PNG)

```py
import numpy as np
import torchvision.datasets as datasets
from utils import set_seed

set_seed(2022)

def load_CIFAR10():
    trainset = datasets.CIFAR10(root='./data', train=True, download=True)
    testset = datasets.CIFAR10(root='./data', train=False, download=True)

    X = np.concatenate((trainset.data, testset.data), axis=0)
    y = np.concatenate((trainset.targets, testset.targets), axis=0)
    
    return X, y

X, y = load_CIFAR10()

print(f'X shape : {X.shape} | y shape : {y.shape} | data type : {type(X)}')
```

X shape : (60000, 32, 32, 3) | y shape : (60000,) | data type : <class 'numpy.ndarray'>

---

### 이상탐지를 위한 데이터셋 구성

#### 데이터 분할

CIFAR-10 데이터는 이상탐지를 위한 데이터가 아닌 이미지 분류를 위한 데이터이다. 따라서 임의로 이상탐지 상황을 가정해야 한다.

CIFAR-10의 클래스는 크게 생물과 이동수단으로 아래처럼 분류할 수 있다.

- 생물 클래스 : bird, cat, deer, dog, frog, horse

- 이동 수단 클래스 : airplane, automobile, ship, truck

이번 튜토리얼에서는 생물 클래스 6개를 정상으로 이동 수단 클래스 중 ship 클래스를 이상으로 정의하고 아래와 같이 학습/검증/테스트 집합으로 분할한다.

- 정상 클래스(36000개) : bird, cat, deer, dog, frog, horse | 이상 클래스(6000개) : ship

- 정상 데이터로만 구성된 학습 집합(정상 데이터 22000개)

- threshold 설정을 위한 검증 집합(정상 데이터 7000개 + 이상 데이터 3000개)

- 이상 탐지 성능 평가를 위한 테스트 집합(정상 데이터 7000개 + 이상 데이터 3000개)

```py
import random

# define normal & abnormal classes
normal_classes = [2,3,4,5,6,7] # bird, cat, deer, dog, frog, horse
abnormal_class = 8 # ship

# get normal instance index
normal_idx = []
for cls in normal_classes:
    normal_idx += np.argwhere(y==cls).reshape(-1).tolist()

# get abnormal instance index
abnormal_idx = np.argwhere(y==abnormal_class).reshape(-1).tolist()

# shuffling indices to split data to train/valid/test
random.shuffle(normal_idx)
random.shuffle(abnormal_idx)

# split data to train/valid/test sets
train_idx = normal_idx[:22000]                          # # of normal : 22000
valid_idx = normal_idx[22000:29000]+abnormal_idx[:3000] # # of normal : 7000 | # of abnormal : 3000
test_idx = normal_idx[29000:]+abnormal_idx[3000:]       # # of normal : 7000 | # of abnormal : 3000

X_train, y_train = X[train_idx], y[train_idx]
X_valid, y_valid = X[valid_idx], y[valid_idx]
X_test, y_test = X[test_idx], y[test_idx]

print(f'train / valid / test = {len(train_idx)} / {len(valid_idx)} / {len(test_idx)}')
```

train / valid / test = 22000 / 10000 / 10000

#### Custom Dataset & Data Loader

데이터를 분할했다면 custom dataset을 구성하고 학습/검증/테스트별 data loader를 구축한다. 

custom dataset에서는 이미지 RGB채널을 평균 0.5, 표준편차 0.5로 nomalize하고 tensor로 변환한다.

data loader는 batch size만큼 데이터를 가져오는 generator를 생성하게 된다.

이 때, 학습을 위한 batch size는 256으로 설정했다.

```py
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class Custom_dataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        
        # define transform for preprocessing
        self.transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        self.X, self.y = X, y
            
    def __getitem__(self, idx):
        return {'X':self.transform(self.X[idx]), 'y':self.y[idx]}
    
    def __len__(self):
        return len(self.y)
    
train_set = Custom_dataset(X_train, y_train)
valid_set = Custom_dataset(X_valid, y_valid)
test_set = Custom_dataset(X_test, y_test)
```

---

### 오토인코더 구축

다음으로 오토인코더 모델 코드를 작성한다. 모델은 이미지 데이터에 많이 활용되는 convolutional autoencoder를 활용한다.

![image](https://user-images.githubusercontent.com/80674834/201905679-7eb4e0e7-ab40-4fee-bc15-605bfa4d47f5.png)

#### 인코더

인코더는 convolution - batchnorm - ReLU - maxpool을 하나의 CNN block으로 구성하여 총 4개의 CNN block를 연결한다. 

최종 latent vector를 생성할 때는 연산의 효율성을 위해서 linear layer 대신에 global average pooling(GAP)을 활용한다.

이미지는 3x32x32 > 32x16x16 -> 64x8x8 -> 128x4x4 -> 256x2x2 -> 256x1x1 순으로 변환되어 최종적으로 256차원의 1d vector로 요약된다.

```py
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, feature_maps=[32,64,128,256]):
        super().__init__()
        
        self.enc = self.make_layers(feature_maps)
        self.gap = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        """x : batch, C, W, H"""

        x = self.enc(x) # B, 256, 2, 2
        x = self.gap(x) # B, 256, 1, 1
        
        return x
    
    def make_layers(self, feature_maps):
        layers = []
        input_dim = 3
        for i, feature_dim in enumerate(feature_maps):

            layers += [nn.Conv2d(input_dim, feature_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(feature_dim),
                    nn.ReLU(True),
                    nn.MaxPool2d((3,3),2)]

            input_dim = feature_dim
        return nn.Sequential(*layers)
```

#### 디코더

디코더는 ConvTranspose - batchnorm - ReLU로 구성된 CNN block이 5개로 이루어진 구조를 가진다.

인코더보다 CNN block이 하나 더 많은 이유는 GAP로 축소된 차원을 다시 복원하기 위함이다.



hid_dim에서 시작해서 layer_dims의 역순으로 점점 커지는 구조를 가진다.

make_layers 함수에서 layer_dims를 역순으로 바꾸고 encoder와 같이 linear layer와 ReLU 함수 시퀀스를 생성한다.

마지막으로 이미지 데이터 공간인 3*32*32차원으로 이미지를 매핑해주는 linear layer를 추가한 뒤, 3by32by32 형태로 reshape을 진행한다.

```py
class Decoder(nn.Module):
    def __init__(self, feature_maps=[256,128,64,32,3]):
        super().__init__()
                
        self.dec = self.make_layers(feature_maps)
    
    def forward(self, x):
        """x : batch, hid dim, 1, 1"""
        x = self.dec(x) # B, 3, 32, 32
        return x
    
    def make_layers(self, feature_maps):
        layers = []
        input_dim = 256
        for i, feature_dim in enumerate(feature_maps):
            if i+1 != len(feature_maps):
                layers += [nn.ConvTranspose2d(input_dim, feature_dim, kernel_size=2, stride=2),
                        nn.BatchNorm2d(feature_dim),
                        nn.ReLU(True)]
            else:
                layers += [nn.ConvTranspose2d(input_dim, feature_dim, kernel_size=2, stride=2)]
            input_dim = feature_dim
        return nn.Sequential(*layers)
```

#### Convolutional Autoencoder

앞에서 작성한 인코더를 하나로 합쳐서 최종적인 Convolutional Autoencoder를 생성한다. 

이 때, 원활한 학습을 위해서 he initialization으로 모델 파라미터를 초기화한다.

```py
from utils import initialize_weights

class Autoencoder(nn.Module):
    def __init__(self,
                 enc_feature_maps=[32,64,128,256],
                 dec_feature_maps=[256,128,64,32,3]):
        super().__init__()
                
        self.enc = Encoder(feature_maps=enc_feature_maps)
        self.dec = Decoder(feature_maps=dec_feature_maps)
        
        self.apply(initialize_weights)
    
    def forward(self, x):
        """x: B,3,32,32"""
        x = self.enc(x) # B, hid dim, 1, 1
        x = self.dec(x) # B, 3, 32, 32
        return x

model = Autoencoder().to(device)
```

---

### 오토인코더 기반 이상탐지 프레임워크

오토인코더 기반 이상탐지 프레임워크는 크게 학습 단계, 이상 탐지 단계로 나눌 수 있다. 각 단계에 맞게 코드를 구성한다.

#### 학습 단계

학습 단계에서는 아래 그림처럼 정상 데이터만을 통해 오토인코더를 학습한다.

![image](https://user-images.githubusercontent.com/80674834/201922823-d37bd13f-fc5f-436e-ac08-0d30e07f7798.png)

총 50 epoch 학습하였으며, 손실함수는 그림과 같이 MSE loss이다. 또한, optimizer는 Adam을 learning rate = 3e-4, weight decay = 5e-4로 설정하여 사용한다.

```py
import time

def train(train_loader, model, loss_fn, opt, epochs = 50):
    
    total_loss = []
    
    for epoch in range(1, epochs):
        model.train()
        train_loss = 0.0
        since = time.time()
        
        for i, batch in enumerate(train_loader):
            x = batch['X'].to(device) # B, 3, 32, 32
            pred = model(x) # B, 3, 32, 32
            
            loss = loss_fn(pred, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            train_loss += loss.item()
        
        train_loss = train_loss/(i+1)
        total_loss.append(train_loss)
        t_time = time.time() - since
    
        print(f'Epoch : {epoch} | train loss : {round(train_loss, 6)} | required time : {round(t_time,2)}')

    return total_loss
    
epochs = 5
loss_fn = nn.MSELoss().to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-4)

total_loss = train(train_loader, model, loss_fn, opt)
```
Epoch : 1 | train loss : 1.34821 | required time : 3.93

Epoch : 2 | train loss : 0.464002 | required time : 1.42

Epoch : 3 | train loss : 0.181021 | required time : 1.41

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ㆍ

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ㆍ

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ㆍ

Epoch : 48 | train loss : 0.027556 | required time : 1.48

Epoch : 49 | train loss : 0.027684 | required time : 1.42

Epoch : 50 | train loss : 0.027613 | required time : 1.5

total loss를 활용해서 loss plot을 그려본다.

'''py
import matplotlib
'''




