## SVM 튜토리얼

#### > ([SVM 설명 보러가기](https://github.com/Saerin-Lim/Business_Analytics/blob/master/2.kernel%20based%20learning/SVM_slide.pdf)) <

이번 튜토리얼에서는 여러 데이터셋을 통해서 평균적으로 가장 좋은 성능을 가지는 kernel을 찾고 SVM을 적용하는 상황에서 가이드라인을 주는 것을 목표로 한다.

일반적으로 많이 알려져있는 linear kernel, polynomial kernel, sigmoid kernel 그리고 RBF kernel을 비교 kernel들로 선정하였다.

실험을 위해 정형 데이터 11개, 비정형 이미지 데이터 4개로 총 15개의 데이터셋을 활용하였다. 아래는 데이터셋 목록이다.

정형 데이터: 데이터명(소스)

1. iris(sklean)
2. breast cancer(sklean)
3. wine(sklean)
4. penguins(seaborn)
5. titanic(seaborn)
6. algerian forest fires(uci repository)
7. breast cancer coimbra(uci repository)
8. heart failure clinical records(uci repository)
9. seeds(uci repository)
10. HCV(uci repository)
11. breast tissue(uci repository)

비정형 데이터: 데이터명(소스)

12. digits(sklearn)
13. MNIST(pytorch)
14. Fashion MNIST(pytorch)
15. Kuzushiji MNIST(pytorch)

---

### 데이터셋 불러오기

가장 먼저 데이터셋을 불러와 dictionary 형태로 저장한다.

데이터를 불러오는 과정에서 간단한 전처리를 진행하고 numpy ndarray형태로 변환한다.

1. 결측치 제거

2. 범주형 변수 one-hot encoding

3. target 변수 분리 및 one-hot encoding

4. scaling(정형 데이터 : standard scaling, 비정형 데이터 : minmax scaling)

데이터셋이 여러 개이기 때문에 모든 데이터를 주피터노트북으로 불러오면 코드가 너무 길어져 utils.py의 load_data함수를 따로 만들었다.

아래의 코드를 통해 데이터를 불러와 datasets이라는 dictionary에 저장하고 df_info에 간단한 데이터 정보를 저장하고 출력한다.

```py
import pandas as pd
import numpy as np
import tabulate
from utils import load_data

data_list = ['iris','breast_cancer','wine','penguins','titanic','algerian_forest_fires',
            'breast_cancer_coimbra','heart_failure_clinical_records','seeds','HCV',
            'breast_tissue','digits','MNIST','FashionMNIST','KMNIST']

datasets = dict()
df_info = pd.DataFrame()

num_samples, num_classes = [], []
for data in data_list:
    datasets[data] = load_data(data)
    n = datasets[data]['x'].shape[0]
    classes = np.unique(datasets[data]['y']).shape[0]
    num_samples.append(n)
    num_classes.append(classes)

df_info['Dataset'] = data_list
df_info['Number of Samples'] = num_samples
df_info['Number of Classes'] = num_classes
```

|    | Dataset                        |   Number of Samples |   Number of Classes |
|:--:|:------------------------------:|:-------------------:|:-------------------:|
|  0 | iris                           |                 150 |                   3 |
|  1 | breast_cancer                  |                 569 |                   2 |
|  2 | wine                           |                 178 |                   3 |
|  3 | penguins                       |                 333 |                   3 |
|  4 | titanic                        |                 182 |                   2 |
|  5 | algerian_forest_fires          |                 244 |                   2 |
|  6 | breast_cancer_coimbra          |                 116 |                   2 |
|  7 | heart_failure_clinical_records |                 299 |                   2 |
|  8 | seeds                          |                 210 |                   3 |
|  9 | HCV                            |                 589 |                   5 |
| 10 | breast_tissue                  |                 106 |                   6 |
| 11 | digits                         |                1797 |                  10 |
| 12 | MNIST                          |               60000 |                  10 |
| 13 | FashionMNIST                   |               60000 |                  10 |
| 14 | KMNIST                         |               60000 |                  10 |


---

### 실험 설계

kernel별 SVM 성능 평가를 위해서 각 데이터셋마다 10회 반복실험을 진행한다.

각각의 실험에서는 학습/테스트 데이터 분할과 시드가 달라지게 된다.

SVM 성능은 정확도를 통해 평가한다.

반복실험을 하기 위한 SVM_experiments를 아래와 같이 만들었다.

이 함수는 데이터와 svm hyperparameter를 입력하면 10회 반복실험을 진행하고 정확도의 평균과 표준편차를 반환한다.

```py
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from utils import set_seed

def SVM_experiments(data, svm_configs):
    seed_list = [0,1,2,3,4,5,6,7,8,9]
    acc_list = []

    for seed in seed_list:
        set_seed(seed)
        
        X_train, X_test, y_train, y_test = train_test_split(data['x'], data['y'], 
                                                            test_size=0.33,
                                                            random_state=seed)
        
        svm = SVC(**svm_configs, random_state=seed)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        
        acc_list.append(accuracy_score(y_test, pred)*100)
    
    return round(np.mean(acc_list),2), round(np.std(acc_list),2), acc_list
```

---

### 실험결과


