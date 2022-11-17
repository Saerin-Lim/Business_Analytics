## autoencoder-based anomaly detection 튜토리얼

#### > [autoencoder-based anomaly detection](https://github.com/Saerin-Lim/Business_Analytics/blob/master/3.anomaly%20detection/model-based%20anomaly%20detection%20slide.pdf) 설명 보러가기 <

이번 튜토리얼에서는 이미지 데이터에 대해서 오토인코더 기반 이상 탐지 프레임워크를 from scratch로 작성하는 것을 목표로 한다.

특히 이상치 탐지는 threshold에 따라서 이상 탐지 성능이 좌우되기 때문에 적절한 threshold를 찾는 것이 매우 중요하다.

일반적으로 k-sigma 방법이나 도메인 지식을 활용한 설정 방법이 있겠지만 이번 튜토리얼에서는 검증 집합에서 grid search를 통해 best threshold를 찾는 방법을 활용한다.


---

### 데이터셋 불러오기

이번 튜토리얼에서 활용할 데이터셋은 대표적인 이미지 데이터인 MNIST와 히라가나 버전 MNIST인 K-MNIST를 활용한다.

두 데이터셋은 각각 28by28 size gray scale image가 총 10개의 class로 나누어져 있다. 전체 이미지 수는 각 데이터마다 70000장 씩 존재한다.

![image](https://user-images.githubusercontent.com/80674834/202104231-01c42651-08b9-4e61-8dcd-9d30ee99157e.png)

```py
import torch
import torchvision.datasets as datasets
from utils import set_seed

set_seed(2022) # fix random number generator seed
device = 'cuda:0' # use gpu. if you want to use cpu, assign like that -> device = 'cpu'

def load_MNIST():
    trainset = datasets.MNIST(root='./data', train=True, download=False)
    testset = datasets.MNIST(root='./data', train=False, download=False)

    X = torch.cat([trainset.data, testset.data], dim=0)
    
    return X

def load_KMNIST():
    trainset = datasets.KMNIST(root='./data', train=True, download=False)
    testset = datasets.KMNIST(root='./data', train=False, download=False)

    X = torch.cat([trainset.data, testset.data], dim=0)
    
    return X

M_X = load_MNIST()
K_X = load_KMNIST()

print(f'MNIST data shape : {M_X.shape} | K-MNIST data shape : {K_X.shape}')
```

MNIST data shape : torch.Size([70000, 28, 28]) | K-MNIST data shape : torch.Size([70000, 28, 28])

---

### 이상탐지를 위한 데이터셋 구성

#### 데이터 분할

MNIST와 K-MNIST는 이상탐지를 위한 데이터가 아닌 이미지 분류를 위한 데이터이다. 따라서 임의로 이상탐지 상황을 가정해야 한다.

본 튜토리얼에서는 MNIST를 정상 데이터로, K-MNIST를 이상 데이터로 정의하고 아래와 같이 학습/검증/테스트 집합으로 분할한다.

이 때, 빠른학습을 위해서 MNIST는 30000장을, K-MNIST는 10000장을 무작위로 샘플링하여 총 40000장의 데이터로 튜토리얼을 진행한다.

- 정상 클래스(30000개) : MNIST | 이상 클래스(10000개) : K-MNIST

- 정상 데이터로만 구성된 학습 집합(정상 데이터 20000개)

- threshold 설정을 위한 검증 집합(정상 데이터 5000개 + 이상 데이터 5000개)

- 이상 탐지 성능 평가를 위한 테스트 집합(정상 데이터 5000개 + 이상 데이터 5000개)

```py
import random
import numpy as np

# randomly sampling img indices
m_idx = np.random.choice(60000, size=(30000,), replace=False) # total normal data   : 30000
k_idx = np.random.choice(60000, size=(10000,), replace=False) # total abnormal data : 10000

# shuffling indices to split data to train/valid/test
random.shuffle(m_idx)
random.shuffle(k_idx)

# sampling normal & abnormal data
normal_X = M_X[m_idx]
abnormal_X = K_X[k_idx]

# generate targets, if normal -> 0 elif abnormal -> 1
normal_y = torch.LongTensor([0]*30000)
abnormal_y = torch.LongTensor([1]*10000)

# shuffling indices to split data to train/valid/test
random.shuffle(m_idx)
random.shuffle(k_idx)

# split data to train/valid/test sets
# train set -> # of normal : 20000
X_train, y_train = normal_X[:20000], normal_y[:20000]

# valid set -> # of normal : 5000 | # of abnormal : 5000
X_valid = torch.cat([normal_X[20000:25000],abnormal_X[:5000]])
y_valid = torch.cat([normal_y[20000:25000],abnormal_y[:5000]])

# valid set -> # of normal : 5000 | # of abnormal : 5000
X_test = torch.cat([normal_X[25000:],abnormal_X[5000:]])
y_test = torch.cat([normal_y[25000:],abnormal_y[5000:]])

print(f'train / valid / test = {len(X_train)} / {len(X_valid)} / {len(X_test)}')
```

train / valid / test = 20000 / 10000 / 10000

#### Custom Dataset & Data Loader

데이터를 분할했다면 custom dataset을 구성하고 학습/검증/테스트별 data loader를 구축한다. 

custom dataset에서는 image size를 (32,32)로 변경한 후, RGB채널 값을 0~1사이로 scaling하고 tensor로 변환한다.

image size를 28by28에서 32by32로 변경한 이유는 convolutional autoencoder를 활용할 때, 크기를 쉽게 맞추기 위함이다.

data loader는 batch size만큼 데이터를 가져오는 generator를 생성하게 된다.

이 때, 학습을 위한 batch size는 256으로 설정했다.

```py
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class Custom_dataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        
        #transform img size (28,28) to (32,32)
        self.transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(32),
                                             transforms.ToTensor()])
        self.X, self.y = X, y
            
    def __getitem__(self, idx):
        return {'X':self.transform(self.X[idx]), 'y':self.y[idx]}
    
    def __len__(self):
        return len(self.y)
    
train_set = Custom_dataset(X_train, y_train)
valid_set = Custom_dataset(X_valid, y_valid)
test_set = Custom_dataset(X_test, y_test)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=256, shuffle=False)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
```

---

### 오토인코더 구축

다음으로 오토인코더 모델 코드를 작성한다. 모델은 이미지 데이터에 많이 활용되는 convolutional autoencoder를 활용한다.

![image](https://user-images.githubusercontent.com/80674834/201905679-7eb4e0e7-ab40-4fee-bc15-605bfa4d47f5.png)

#### 인코더

인코더는 convolution - batchnorm - ReLU - maxpool을 하나의 CNN block으로 구성하여 총 3개의 CNN block를 연결한다. 

최종 latent vector를 생성할 때는 연산의 효율성을 위해서 linear layer 대신에 global average pooling(GAP)을 활용한다.

이미지는 1x32x32 > 16x16x16 -> 32x8x8 -> 64x4x4 -> 128x2x2 -> 128x1x1 순으로 변환되어 최종적으로 128차원의 1d vector로 요약된다.

```py
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, feature_maps=[16,32,64,128]):
        super().__init__()
        
        self.enc = self.make_layers(feature_maps)
        self.gap = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        """x : batch, 1, W, H"""

        x = self.enc(x) # B, 128, 2, 2
        x = self.gap(x) # B, 128, 1, 1
        
        return x
    
    def make_layers(self, feature_maps):
        layers = []
        input_dim = 1
        for i, feature_dim in enumerate(feature_maps):

            layers += [nn.Conv2d(input_dim, feature_dim, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(feature_dim),
                    nn.ReLU(True),
                    nn.MaxPool2d((2,2),2)]

            input_dim = feature_dim
        return nn.Sequential(*layers)
```

#### 디코더

디코더는 ConvTranspose - batchnorm - ReLU로 구성된 CNN block이 5개로 이루어진 구조를 가진다.

인코더보다 CNN block이 하나 더 많은 이유는 GAP로 축소된 차원을 다시 복원하기 위함이다.

출력층 CNN block에서는 활성화 함수를 사용하지 않았다.

```py
class Decoder(nn.Module):
    def __init__(self, feature_maps=[128,64,32,16,1]):
        super().__init__()
                
        self.dec = self.make_layers(feature_maps)
    
    def forward(self, x):
        """x : batch, hid dim, 1, 1"""
        x = self.dec(x) # B, 1, 32, 32
        return x
    
    def make_layers(self, feature_maps):
        layers = []
        input_dim = 128
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
                 enc_feature_maps=[16,32,64,128],
                 dec_feature_maps=[128,64,32,16,1]):
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

총 50 epoch 학습하였으며, 손실함수는 그림과 같이 MSE loss이다. 또한, optimizer는 Adam을 learning rate = 1e-3, weight decay = 5e-4로 설정하여 사용한다.

```py
import time

def train(train_loader, model, loss_fn, opt, epochs = 50):
    
    total_loss = []
    
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        since = time.time()
        
        for i, batch in enumerate(train_loader):
            x = batch['X'].to(device) # B, 1, 32, 32
            pred = model(x) # B, 1, 32, 32
            
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
    
epochs = 50
loss_fn = nn.MSELoss().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

total_loss = train(train_loader, model, loss_fn, opt, epochs=epochs)
```

Epoch : 1 | train loss : 0.783915 | required time : 5.11

Epoch : 2 | train loss : 0.084003 | required time : 1.66

Epoch : 3 | train loss : 0.039879 | required time : 1.8

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ㆍ

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ㆍ

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ㆍ

Epoch : 48 | train loss : 0.004214 | required time : 3.79

Epoch : 49 | train loss : 0.004003 | required time : 3.8

Epoch : 50 | train loss : 0.004074 | required time : 3.14

total loss를 활용해서 loss plot을 그려본다. 아래의 loss plot을 통해서 학습이 잘 이루어졌음을 확인할 수 있다.

```py
import matplotlib.pyplot as plt
import seaborn as sns

sns.lineplot(x=range(6,epochs+1), y=total_loss[5:])
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Train Loss per Epoch')
plt.show()
```

![loss](https://user-images.githubusercontent.com/80674834/202121359-85a24ef5-6f40-4ec6-90c6-f56c001dd8c1.png)

다음으로 원본 이미지와 복원된 이미지를 시각화하여 학습이 잘 이루어졌는지 정성적으로 확인한다. 실제로 오토인코더가 원본 이미지를 어느정도 잘 복원하는 것을 볼 수 있다.

```py
# get original imgs
batch = next(iter(train_loader))
original = batch['X'][:2].to(device) # 2, 3, 32, 32

# get reconstruction imgs
with torch.no_grad():
    pred = model(original) # 2, 3, 32, 32

# transform tensor to numpy with shape 2, 32, 32, 3
original = np.transpose(original.detach().cpu().numpy(), (0, 2, 3, 1))
pred = np.transpose(pred.detach().cpu().numpy(), (0, 2, 3, 1))

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(original[0], cmap='gray')
plt.title('Orignal Image 1')

plt.subplot(2,2,2)
plt.imshow(pred[0], cmap='gray')
plt.title('Reconstruction Image 1')

plt.subplot(2,2,3)
plt.imshow(original[1], cmap='gray')
plt.title('Orignal Image 2')

plt.subplot(2,2,4)
plt.imshow(pred[1], cmap='gray')
plt.title('Reconstruction Image 2')
```

![train_visualized](https://user-images.githubusercontent.com/80674834/202121392-1a29529d-eb3a-4456-b8ff-679fe3f45aba.png)

---

#### 이상 탐지 단계

이상 탐지 단계에서는 학습이 완료된 오토인코더에 새로운 이미지를 넣어서 복원값을 구한 뒤, 실제값과 복원값의 차이를 통해 anomaly score를 정의하게 된다.

일반적으로 anomaly score는 MSE와 MAE가 많이 활용되며 튜토리얼에서는 MSE를 활용한다.

새로운 이미지에 대한 anomaly score가 계산되면 threshold와 비교해서 threshold보다 크면 이상, 작으면 정상으로 분류한다.

![image](https://user-images.githubusercontent.com/80674834/201941533-4f548c93-7bb2-4131-b705-6475a980a0b5.png)

먼저  anomaly score를 계산하는 함수와 최종적으로 detection을 해서 탐지 성능을 반환하는 함수를 작성한다.

```py
import pandas as pd

def get_anomaly_score(model, data_loader):
    model.eval()
    loss_fn = nn.MSELoss(reduction='none').to(device)
    anomaly_scores = []
    targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            x = batch['X'].to(device) # B, 1, 32, 32
            targets += batch['y'].detach().tolist() # B

            pred = model(x) # B, 1, 32, 32
            mse = loss_fn(pred, x) # B, 1, 32, 32
            anomaly_scores += torch.mean(mse.view(-1,32*32),
                                       dim=1).detach().cpu().tolist() # B

    
    df = pd.DataFrame({'img id':range(1,len(targets)+1),
                       'anomaly score':anomaly_scores,
                       'target':targets})
    
    return df
    
def anomaly_detection(df:pd.DataFrame, threshold:float):
    df['pred'] = np.where(df['anomaly score'] > threshold, 1, 0)
    
    return f1_score(df['target'], df['pred'])
```

get_anomaly_score 함수는 아래와 같이 모델과 anomaly score를 계산할 data를 dataloader형태로 받아서 anomaly score와 target이 있는 dataframe을 반환한다.

|    |   img id |   anomaly score |   target |
|:--:|:--------:|:---------------:|:--------:|
|  0 |        1 |       0.012854  |        0 |
|  1 |        2 |       0.018402  |        0 |
|  2 |        3 |       0.0105461 |        0 |
|  3 |        4 |       0.0100898 |        0 |
|  4 |        5 |       0.0155868 |        0 |

anomaly_detection 함수는 get_anomaly_score에서 나온 dataframe과 threshold를 입력받아서 anomaly score와 threshold를 비교해 정상/이상을 분류한다.

그리고 최종적으로 예측한 정상/이상과 실제 정상/이상에 대한 탐지 성능를 반환한다. 본 튜토리얼에서는 f1-score를 모델 성능 평가 지표로 활용하였다.

먼저 get_anomaly_score 함수를 이용해서 검증집합에 대한 anomaly score를 계산하고 정상과 이상의 anomaly score 통계량 및 분포를 확인한다.

확인한 통계량을 바탕으로 threshold grid search space를 결정할 것이다.

```py
# get anomaly scores of valid set
df = get_anomaly_score(model, valid_loader)

# check statistics of anomlay scores according to class(normal or abnormal)
normal_df = df[df['target']==0]['anomaly score']
abnormal_df = df[df['target']==1]['anomaly score']

print(f'normal mean & std : ({round(normal_df.mean(),6)},{round(normal_df.std(),6)})')
print(f'normal max & min : ({round(normal_df.max(),6)},{round(normal_df.min(),6)}) \n')

print(f'abnormal mean & std : ({round(abnormal_df.mean(),6)},{round(abnormal_df.std(),6)})')
print(f'abnormal max & min : ({round(abnormal_df.max(),6)},{round(abnormal_df.min(),6)}) \n')

# histogram of anomlay scores according to class(normal or abnormal)
sns.histplot(df[df['target']==0]['anomaly score'],
             color='blue', label='normal')

sns.histplot(df[df['target']==1]['anomaly score'],
             color='red', label='abnormal')
plt.title('Histogram of Anomaly Scores')
plt.legend()
```

normal mean & std : (0.011799,0.003691)
normal max & min : (0.027663,0.002723) 

abnormal mean & std : (0.052069,0.024842)
abnormal max & min : (0.226932,0.006235) 

![hist](https://user-images.githubusercontent.com/80674834/202124111-80c6bfd7-580d-426b-9048-a3c8fd6547e9.png)

먼저 히스토그램을 보면 이상 데이터인 경우 anomaly score가 대부분 높은 것을 확인할 수 있다. 즉, 모델이 검증 데이터에 대해 이상탐지를 잘 할 수 있음을 보여준다.

다음으로 정상 데이터의 최대값과 이상 데이터의 최소값을 보면 0.027663과 0.006235이다.

따라서 threshold grid search space를 [0.006, 0.028]까지로 설정한다.

그리고 0.006부터 0.0005씩 증가시키며 검증 데이터에 대한 이상 탐지 성능을 평가한 후, 가장 좋은 성능을 보인 threshold를 최종 threshold로 결정한다.

* threshold grid search
```py
def threshold_search(df:pd.DataFrame, min, max, step):
    search_space = np.arange(min, max+step, step)
    best_f1 = 0.
    best_acc = 0.
    best_threshold = 0.
    
    for threshold in search_space:
        crt_f1, crt_acc = anomaly_detection(df, threshold)
        
        if crt_f1 > best_f1:
            best_f1 = crt_f1
            best_threshold = threshold
            best_acc = crt_acc
    
    return round(best_f1*100,2), round(best_acc*100,2), round(best_threshold, 5)

best_f1, best_acc, best_threshold = threshold_search(df, 0.006, 0.028, 0.0005)

print(f'best f1-score : {best_f1} | best accuracy : {best_acc} | best threshold : {best_threshold}')
```

best f1-score : 96.58 | best accuracy : 96.61 | best threshold : 0.0195

Grid search 결과, best threshold = 0.0195이며 검증 데이터에 대한 f1-score와 accuracy는 각각 96.58%와 96.61%로 매우 좋은 것을 확인 할 수 있다.

#### 최종 이상 탐지 결과

마지막으로 grid search를 통해 찾은 best threshold를 통해 테스트 집합에 대한 이상 탐지를 실시하고, 실제로 best threshold일 때 성능이 가장 좋은지 확인한다.

```py
# get anomaly scores of test set
df = get_anomaly_score(model, test_loader)

# histogram of anomlay scores according to class(normal or abnormal)
sns.histplot(df[df['target']==0]['anomaly score'],
             color='blue', label='normal')

sns.histplot(df[df['target']==1]['anomaly score'],
             color='red', label='abnormal')
plt.title('Histogram of Anomaly Scores')
plt.legend()

# anomaly detection
f1, acc = anomaly_detection(df, best_threshold)

print(f'Test f1-score : {round(f1*100,2)} | Test accuracy : {round(acc*100,2)}')
```
Test f1-score : 96.54 | Test accuracy : 96.57

![test results](https://user-images.githubusercontent.com/80674834/202137230-dcf62748-7db7-4c15-95fd-6461b285a066.png)

위 그래프는 검증 집합에서 grid search로 찾은 best threshold가 실제 테스트 집합에서도 거의 가장 좋은 threshold임을 보여주고 있다.

---

### 결론

이번 튜토리얼에서는 오토인코더 기반 이상탐지 프레임워크를 from scrath로 구현해 보았다.

그리고 threshold는 검증 집합에 대한 grid search를 통해 찾고 실제 테스트 집합에서도 잘 적용이 된다는 것을 확인하였다.

하지만 실험에서 활용한 데이터는 이상 데이터가 매우 많이 존재해서 검증 집합을 만들기 충분한 상황이었고, 검증 집합과 테스트 집합의 데이터 분포가 비슷한 상황이다.

현실 상황에서는 이상 데이터가 매우 적을 뿐만 아니라 이상 데이터의 분포가 상이할 가능성도 존재한다.

따라서 위 방법은 이상 데이터가 충분해서 검증 데이터로 분할이 가능하고, 대부분의 이상 분포를 표현할 수 있는 경우 사용하는 것을 추천한다.
