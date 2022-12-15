
## Self-training 튜토리얼

Self-training은 간단한 semi-supervised learning 방법론 중 하나로 pseudo labeling을 이용한 방법이다.

Self-training은 unlabeled data에 대한 추론을 한 뒤, 특정 클래스에 속할 확률이 threshold보다 높으면 pseudo label로 활용한다.

전체적인 알고리즘을 아래와 같다.

1. labeled data로 모델을 학습

2. 학습된 모델을 통해서 unlabeled data 추론

3. 추론 확률이 threshold를 넘는 unalbeled data에 대해 pseudo labeling 진행

4. labeled data와 pseudo-labeled data를 통해서 새로운 모델 학습

5. unlabeled data에 모두 pseudo-label이 부여되거나 종료 조건을 만족하면 알고리즘 종료

![image](https://user-images.githubusercontent.com/80674834/207237944-07a86aae-d30a-4b90-9ac0-1a834ec0f61c.png)

이 과정에서 pseudo labeling을 할 unlabeled data를 정하는 전략은 중요한 hyperparameter 중 하나이다.

일반적으로 확률값이 특정 threshold 이상 (ex.0.95 이상)인 unlabeled data에 대해 pseudo labeling을 하거나, 확률값이 높은 top-k unlabeled data를 뽑는 방법이 있으며, 두가지를 동시에 사용하는 방법도 있다.

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
        self.seed = 2022                         # seed
        self.device = 'cuda:0'
        
        # data hyperparameters
        self.train_imgs = 50000                  # number of train imgs
        self.test_imgs = 10000                   # number of test imgs
        self.num_l = 25000                       # number of labeled data
        self.num_u = self.train_imgs - self.num_l# number of unlabeled data
        
        # train hyperparameters
        self.batch_size = 128                    # batch size
        self.lr = 3e-4                           # learning rate
        self.weight_decay = 5e-4                 # weight decay
        self.epochs = 5                          # train epoch
        
        # self train hyperparameters
        self.iteration = 3                       # self training iteration
        #self.period = 10                        # pseudo labeling period
        
        # pseudo label strategy hyperparameters
        self.strategy = 'threshold'              # pseudo labeling strategy ['threshold', 'top_k', 'both']
        self.threshold = 0.95                    # threshold for pseudo labeling
        self.top_k = 2500                        # top-k for pseudo labeling
```

---

### Custom Dataset 구축 (datasets.py)

CIFAR-10은 pytorch에 내장되어 있어 쉽게 dataset을 구축할 수 있다. 하지만 self-training을 위해서는 unlabeled data의 index가 있어야지만 unlabeled data에서 pseudo label을 할 data를 식별 할 수 있다. 

따라서 아래와 같이 custom Dataset을 작성했다. transform을 통해 이미지를 nomalize하였고, getitem에서 이미지의 index까지 반환하도록 작성하였다.

```py
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
```

### Inital Dataloader & Update Dataloader(dataloaders.py)

CIFAR-10은 기본적으로 학습 데이터와 테스트 데이터가 50000장/10000장으로 분할되어 있다. 여기에 semi-supervised scenario를 만들기 위해서 학습 데이터 중 특정 개수만큼 labeled dataset과 unlabeled dataset을 만들어야 한다.

또한, self-training iteration 과정에서 생성되는 pseudo label과 해당하는 이미지는 labeled dataset으로 옮겨짐과 동시에 unlabeled dataset에서 삭제되어야 한다.

이 과정에서 init_dataloaders 함수와 update_dataloaders 함수를 작성하였다.

먼저 init_dataloaders 함수는 학습을 시작하기 전, dataloader를 초기화 한다. 즉, 전체 학습 데이터에서 labeled dataset과 unlabeled dataset을 특정 개수만큼(args.num_l) 분할하고 각각의 loader를 정의한다.

다음으로 update_dataloaders 함수는 pseudo label과 해당하는 이미지 indices를 입력받아 pseudo dataset을 만들고, labeled dataset과 concat하여 new_labeled_dataset을 만들게 된다.

동시에 unlabeled dataset에서 pseudo label이 된 이미지 indices를 제거하여, labeled loader와 unlabeled loader를 업데이트한다.

```py
import random
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.datasets as torch_datasets

from datasets import CIFAR10_Dataset

def init_dataloaders(args, X_train, y_train, X_test, y_test):
    
    # define label & unlabel data
    total_indices = list(range(args.train_imgs))
    label_indices = random.sample(total_indices, args.num_l)
    unlabel_indices = list(set(total_indices)-set(label_indices))
    
    X_labeled, y_labeled = X_train[label_indices], np.array(y_train)[label_indices].tolist()
    X_unlabeled, y_unlabeled = X_train[unlabel_indices], np.array(y_train)[unlabel_indices].tolist()
    
    print(f'# of train data : {args.train_imgs} | # of test data : {args.test_imgs}')
    print(f'# of labeled data in trainset : {len(label_indices)} | # of unlabeled data in trainset : {len(unlabel_indices)}')
    print(f'labeled : unlabeled = {round(len(label_indices)/500,2)}% : {100-round(len(label_indices)/500,2)}%\n')
    
    # make datasets
    labeled_set = CIFAR10_Dataset(X_labeled, y_labeled)
    unlabeled_set = CIFAR10_Dataset(X_unlabeled, y_unlabeled)
    test_set = CIFAR10_Dataset(X_test, y_test)
    
    # make dataloader
    label_loader = DataLoader(labeled_set, batch_size=args.batch_size, shuffle=True)
    unlabel_loader = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    
    loaders = dict(labeled=label_loader,
                   unlabeled=unlabel_loader,
                   test=test_loader)
    
    indices = dict(total=total_indices,
                   labeled=label_indices,
                   unlabeled=unlabel_indices,
                   pseudo=[])
    
    return loaders, indices

def update_dataloaders(args, X_train, y_train, y_pseudo, 
                       loaders, indices, pseudo_indices, pseudo_labels):
    
    # update pseudo labeled & unlabeled indices
    indices['pseudo'] += pseudo_indices
    indices['unlabeled'] = list(set(indices['unlabeled'])-set(pseudo_indices))
    
    # define labeled & pseudo labeled & unlabeled data
    X_labeled, y_labeled = X_train[indices['labeled']], np.array(y_train)[indices['labeled']].tolist()
    X_unlabeled, y_unlabeled = X_train[indices['unlabeled']], np.array(y_train)[indices['unlabeled']].tolist()
    X_pseudo = X_train[indices['pseudo']]
    y_pseudo += pseudo_labels # update pseudo labels
    
    # make datasets
    labeled_set = CIFAR10_Dataset(X_labeled, y_labeled)
    unlabeled_set = CIFAR10_Dataset(X_unlabeled, y_unlabeled)
    pseudo_set = CIFAR10_Dataset(X_pseudo, y_pseudo)
    
    # concat labeled & pseudo labeled set
    if len(indices['pseudo']) != 0:
        new_labeled_set = ConcatDataset([labeled_set, pseudo_set])
    else:
        new_labeled_set = labeled_set

    # make new dataloader
    loaders['labeled'] = DataLoader(new_labeled_set, batch_size=args.batch_size, shuffle=True)
    loaders['unlabeled'] = DataLoader(unlabeled_set, batch_size=args.batch_size, shuffle=True)
    
    # number of labeled & unlabeled data
    num_pseudo = len(pseudo_indices)
    total_l = len(indices['labeled'])
    total_u = len(indices['unlabeled'])
    
    return num_pseudo, total_l, total_u
```

---

### Resnet18

다음으로 학습 모델을 구축하였다. 이미지 데이터이기 때문에 CNN기반 모델 중 선택했으며, 그 중 가장 많이 활용되는 Resnet18을 선택했다.

코드는 아래와 같으며, [해당 github](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)에서 가지고 왔다.

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

---

### training(Task -> trainer.py)

다음으로 모델 학습을 위한 trainer를 구성하였다. trainer에는 labeled data를 통해 모델을 학습시키는 train 함수,

테스트 데이터에 대해서 모델을 평가하는 test 함수,

마지막으로 학습된 모델을 통해서 unlabeled data에 대한 추론을 하고, pseudo label을 얻는 get_pseudo_label 함수로 나뉜다.

여기서 가장 핵심은 get_pseudo_label 함수로 특정 epoch마다(args.labeling_period) unlabeled data에 대해서 추론을 하고, 전략에 맞춰서 pseudo labeling을 진행한 뒤, 해당하는 unlabeled data에 대한 index와 pseudo label을 반환한다.

```py
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score

def train(args, model, opt, criterion, train_loader):
    model.train()
    softmax = nn.Softmax()
    
    train_loss = 0.
    
    y_true, y_pred = [], []
    
    # supervised learning with labeled data + pseudo labeled data
    for idx, batch in enumerate(train_loader):
        inputs = batch['X'].to(args.device) # B, 3, 32, 32
        targets = batch['y'].to(args.device) # B,
        
        # get pred
        preds = model(inputs) # B, 10
        
        # get cross entropy loss
        loss = criterion(preds, targets)
        
        # backpropagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        train_loss += loss.item()
        
        y_true += targets.detach().cpu().tolist()
        y_pred += softmax(preds).argmax(dim=-1).detach().cpu().tolist()
        
    train_loss /= (idx+1)
    train_acc = round(accuracy_score(y_true, y_pred)*100, 2)
    
    return train_loss, train_acc

def test(args, model, test_loader):
    model.eval()
    softmax = nn.Softmax()
    
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['X'].to(args.device) # B, 3, 32, 32
            
            # get pred
            probs = softmax(model(inputs)) # B, 10
            y_pred += probs.argmax(dim=-1).detach().cpu().tolist() # B,
            
            y_true +=  batch['y'].detach().tolist() # B,
        
    acc = round(accuracy_score(y_true, y_pred)*100, 2)
    
    return acc

def get_pseudo_label(args, model, unlabeled_loaders):
    model.eval()
    softmax = nn.Softmax()
    
    
    data_indices = [] # total unlabeled data index
    max_probs = [] # max prob list
    pred_labels = [] # pseudo label list
    
    # get probability of unlabeled data
    with torch.no_grad():
        for batch in unlabeled_loaders:
            
            data_indices += batch['idx'].tolist()
            inputs = batch['X'].to(args.device) # B, 3, 32, 32
            
            # get class distribution using softmax function
            preds = softmax(model(inputs)) # B, 10
            
            # get max prob and pseudo label
            probs, pseudo = preds.max(dim=1)
            
            max_probs += probs.detach().cpu().tolist()
            pred_labels += pseudo.tolist()
    
    # get pseudo label
    if args.strategy == 'threshold':
        candidates = np.argwhere(np.array(max_probs) > args.threshold).reshape(-1)
        
    elif args.strategy == 'top_k':
        candidates = np.argsort(max_probs)[::-1][:args.top_k]
        
    else:
        candidates_1 = np.argwhere(np.array(max_probs) > args.threshold).reshape(-1)
        candidates_2 = np.argsort(max_probs)[::-1][:args.top_k]
        
        # data taht satisfy both threshold and top_k
        candidates = np.intersect1d(candidates_1, candidates_2)
    
    pseudo_indices = np.array(data_indices)[candidates].tolist()
    pseudo_labels = np.array(pred_labels)[candidates].tolist()
    
    return pseudo_indices, pseudo_labels
```

---

### Main function(run.py)

위에서 작성한 모듈들을 바탕으로 main 코드를 완성한다.

전체적인 코드 과정을 보면 실험을 위해서 시드를 고정한 후, init_dataloaders함수를 통해 labeled & unlabeled loader를 초기화한다.

그리고 사전에 정의한 iteration만큼 모델과 optimizer, loss function(crossentropy 사용)을 정의하고 학습을 시작한다.

각 interation이 끝날 때 마다 get_pseudo_label함수를 통해 pseudo label을 얻고 update_dataoladers함수에 입력하여 dataloader를 업데이트한다.

업데이트 된 dataloader를 통해서 다시 새로운 모델을 학습하고 학습이 완료되면 test함수를 통해서 testset에 대한 추론을 진행하고 accuracy를 반환한다.

종료 조건이 만족 될 때까지 반복한다.

```py
import torch, time

from Utils.utils import *
from dataloaders import *
from Task.trainer import *
from model import ResNet, ResidualBlock

import warnings
warnings.filterwarnings(action='ignore')

def main(args):
    
    # set seed & device
    set_seed(args.seed)
    device = args.device
    total_l = args.num_l
    
    # load CIFAR10
    trainset = torch_datasets.CIFAR10(root='./Data', train=True, download=False)
    testset = torch_datasets.CIFAR10(root='./Data', train=False, download=False)
    
    X_train, y_train = trainset.data, trainset.targets
    X_test, y_test = testset.data, testset.targets
    
    # initialize dataloaders & pseudo labels
    loaders, indices = init_dataloaders(args, X_train, y_train, X_test, y_test)
    y_pseudo = []
    
    # start self-training
    print(f'>>> Self-Training Start...')
    iter_acc = []
    num_label_list = [total_l]
    for i in range(1, args.iteration+1):
        
        # define model, optimizer, criterion
        model = ResNet(ResidualBlock, [2,2,2]).to(device)
        opt = torch.optim.Adam(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        
        ## Train Model
        print(f'>>> {i}th Iteration...')
        for epoch in range(1, args.epochs+1):
            since = time.time()
            train_loss, train_acc = train(args, model, opt, criterion, loaders['labeled'])
            train_time = round(time.time()-since,2)

            print(f'Iteration : {i} | Epoch : {epoch} | Num Labels : {total_l}')
            print(f'train loss : {round(train_loss,4)} | train acc : {train_acc}% | train time : {train_time}sec\n')
            

        pseudo_indices, pseudo_labels = get_pseudo_label(args, model, loaders['unlabeled'])
        num_pseudo, total_l, total_u = update_dataloaders(args, X_train, y_train, y_pseudo, 
                                                          loaders, indices, pseudo_indices, pseudo_labels)

        labeled_rate = round(total_l/(total_l+total_u)*100,2)
        num_label_list.append(total_l)
        print('Pseudo labeling & update DataLoaders')
        print(f'# of new pseudo labels : {num_pseudo} | labeled : unlabeled = {labeled_rate}% : {round(100-labeled_rate,2)}%\n')
    
        print(f'>>> {i}th Iteration Test Start...')
        test_acc = test(args, model, loaders['test'])
        iter_acc.append(test_acc)
        print(f'Test Accuracy : {test_acc}')
    
    acc = round(np.mean(iter_acc),2)
    
    return acc, iter_acc, num_label_list
```

---

### 실험

Self-training iteration = 3으로 설정하고 각 iteration마다 모델 학습 epoch = 50 으로 설정하였다.

Pseudo labeling strategy는 threshold, top-k, 그리고 둘 다 활용하는 both 총 세 가지를 실험하였다.

이 실험에서 threshold와 top-k 설정은 매우 중요하지만 적절한 값을 찾는 시간이 너무 오래 걸리기 때문에 일반적으로 많이 사용하는 threshold=0.95로 설정하였다.

top-k는 전체 학습 데이터 50000개 중 5%에 해당하는 2500으로 설정하였다.

또한, 초기 labeled data 개수에 따라서 성능이 어떻게 변화하는지 확인하기 위해서 labeled data 개수를 500(1%), 2500(5%), 5000(10%), 12500(25%), 25000(50%)로 바꿔가며 실험을 진행하였다.

각 실험 결과는 results_df에 interation 별 label 개수와 accuracy가 저장되며 지도학습 결과는 첫 번째 iteration 결과로 정의하였다.

```py
import pandas as pd
from run import main
from Utils.config import Config

import warnings
warnings.filterwarnings(action='ignore')

results_df = pd.DataFrame(columns=['Seed','Strategy','n_1','n_2','n_3','acc_1','acc_2','acc_3'])

args = Config()

labels_list = [500, 2500, 5000, 12500, 25000]

# self-training with threshold strategy
for num_label in labels_list:
    args.num_l = num_label
    args.strategy = 'threshold'
    acc, iter_acc, label_rate = main(args)
    crt_dict = {'Seed':args.seed,
                'Strategy':args.strategy,
                'n_1':label_rate[0],
                'n_2':label_rate[1],
                'n_3':label_rate[2],
                'acc_1':iter_acc[0],
                'acc_2':iter_acc[1],
                'acc_3':iter_acc[2]}
    results_df = pd.concat([results_df, pd.DataFrame([crt_dict])])
    
# self-training with top-k strategy
for num_label in labels_list:
    args.num_l = num_label
    args.strategy = 'top_k'
    acc, iter_acc, label_rate = main(args)
    crt_dict = {'Seed':args.seed,
                'Strategy':args.strategy,
                'n_1':label_rate[0],
                'n_2':label_rate[1],
                'n_3':label_rate[2],
                'acc_1':iter_acc[0],
                'acc_2':iter_acc[1],
                'acc_3':iter_acc[2]}
    results_df = pd.concat([results_df, pd.DataFrame([crt_dict])])
    
# self-training with both strategy
for num_label in labels_list:
    args.num_l = num_label
    args.strategy = 'both'
    acc, iter_acc, label_rate = main(args)
    crt_dict = {'Seed':args.seed,
                'Strategy':args.strategy,
                'n_1':label_rate[0],
                'n_2':label_rate[1],
                'n_3':label_rate[2],
                'acc_1':iter_acc[0],
                'acc_2':iter_acc[1],
                'acc_3':iter_acc[2]}
    results_df = pd.concat([results_df, pd.DataFrame([crt_dict])])
```

---

### 실험 
 
