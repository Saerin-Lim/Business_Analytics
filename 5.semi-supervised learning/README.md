
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

### hyperparameters for experiments

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

### DataLoaders for semi-supervised scenario

CIFAR-10은 기본적으로 학습 데이터와 테스트 데이터가 50000장/10000장으로 분할되어 있다. 여기에 semi-supervised scenario를 만들기 위해서 학습 데이터 중 특정 개수만큼 labeled dataset과 unlabeled dataset을 만들고 각각 dataloader를 구축해야 한다. 

아래 make_dataloaders 함수는 CIFAR-10 데이터를 불러와서 간단한 전처리(nomalize)를 진행한 뒤, label_loader, unlabel_loader, test_loader를 출력한다.

이 때, labeled set과 unlabeled set을 나누기 위해 SubsetRandomSampler를 이용하는데, 이 함수는 전체 dataset에서 원하는 subset index를 준다면 subset data만 dataloader로 뽑아오게 된다. 

total_indices에서 원하는 개수(args.num_l)만큼 label data index를 무작위로 뽑아 label_indices를 만든 뒤, 나머지를 unlabel_indices로 정의하였다. 

그리고 추후, unlabel_indices의 값을 label_indices로 넘겨줘야 하기 때문에 이 둘 역시 반환받는다.

```py
import random
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

def make_dataloaders(args):
    
    # transform to convert img to tensor and normalize
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
    
    # load dataset
    trainset = datasets.CIFAR10(root='./Data', train=True, download=False, transform=transform)
    testset = datasets.CIFAR10(root='./Data', train=False, download=False, transform=transform)
    
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
    
    return label_loader, unlabel_loader, test_loader, label_indices, unlabel_indices
```

---

### make_dataLoaders for semi-supervised scenario


