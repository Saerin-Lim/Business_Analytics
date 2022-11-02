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

다수의 데이터셋을 주피터노트북으로 불러와 처리하면 코드가 너무 길어져 utils.py의 load_data함수를 따로 만들었다.

그리고 빠른 학습을 위해서 MNIST, FashionMNIST, KMNIST 데이터를 각 클래스별로 150개씩 샘플링하여 60000개의 관측치를 1500개로 줄였다.

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

num_samples, num_classes, num_input = [], [], []
for data in data_list:
    datasets[data] = load_data(data)
    n = datasets[data]['x'].shape[0]
    m = datasets[data]['x'].shape[1]
    classes = np.unique(datasets[data]['y']).shape[0]
    num_samples.append(n)
    num_classes.append(classes)
    num_input.append(m)

df_info['Dataset'] = data_list
df_info['Number of Samples'] = num_samples
df_info['Number of Classes'] = num_classes
df_info['Number of Input Variables'] = num_input
```

|    | Dataset                        |   Number of Samples |   Number of Classes |   Number of Input Variables |
|:--:|:------------------------------:|:-------------------:|:-------------------:|:---------------------------:|
|  0 | iris                           |                 150 |                   3 |                           4 |
|  1 | breast_cancer                  |                 569 |                   2 |                          30 |
|  2 | wine                           |                 178 |                   3 |                          13 |
|  3 | penguins                       |                 333 |                   3 |                           7 |
|  4 | titanic                        |                 182 |                   2 |                          23 |
|  5 | algerian_forest_fires          |                 244 |                   2 |                          10 |
|  6 | breast_cancer_coimbra          |                 116 |                   2 |                           9 |
|  7 | heart_failure_clinical_records |                 299 |                   2 |                          12 |
|  8 | seeds                          |                 210 |                   3 |                           7 |
|  9 | HCV                            |                 589 |                   5 |                          12 |
| 10 | breast_tissue                  |                 106 |                   6 |                           9 |
| 11 | digits                         |                1797 |                  10 |                          64 |
| 12 | MNIST                          |                1500 |                  10 |                         784 |
| 13 | FashionMNIST                   |                1500 |                  10 |                         784 |
| 14 | KMNIST                         |                1500 |                  10 |                         784 |


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

def SVM_experiments(data, kernel, svm_configs={'C':1}):
    seed_list = [0,1,2,3,4,5,6,7,8,9]
    acc_list = []

    for seed in seed_list:
        set_seed(seed)
        
        X_train, X_test, y_train, y_test = train_test_split(data['x'], data['y'], 
                                                            test_size=0.33,
                                                            random_state=seed)
        
        svm = SVC(kernel=kernel, random_state=seed, **svm_configs)
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        
        acc_list.append(accuracy_score(y_test, pred)*100)
    
    return round(np.mean(acc_list),2)
```

SVM의 hyperparameter는 기본적으로 sklearn에서 제공하는 default value를 사용하며,

polynomial kernel에 대해서는 degree를 3,5,7,9,11로 설정한 뒤 가장 좋은 degree의 성능을 최종 성능으로 채택한다.

---

### 실험 및 실험 결과

각 데이터 별로 SVM_experiments함수를 통해 10회 반복실험을 진행하고 평균 정확도를 df_results에 저장한다.

또한, 각 kernel별로 시간을 비교하기 위해서 df_time에 데이터 별 실험 시간을 저장한다.

이 과정을 linear kernel, polynomial kernel, sigmoid kernel, rbf kernel마다 반복 진행한다.

```py
import time

df_results = pd.DataFrame()
df_time = pd.DataFrame()

df_results['Dataset'] = data_list
df_time['Dataset'] = data_list

for kernel in ['linear','poly','sigmoid','rbf']:
    acc_list = []
    time_list = []

    if kernel != 'poly':
        for key, data in datasets.items():
            
            s_time = time.time()
            crt_acc = SVM_experiments(data, kernel)
            r_time = time.time() - s_time
            
            acc_list.append(crt_acc)
            time_list.append(r_time)
    
    else:
        for key, data in datasets.items():
            
            degree_acc = []
            s_time = time.time()
            degree_acc.append(SVM_experiments(data, kernel, {'degree':3}))
            degree_acc.append(SVM_experiments(data, kernel, {'degree':5}))
            degree_acc.append(SVM_experiments(data, kernel, {'degree':7}))
            degree_acc.append(SVM_experiments(data, kernel, {'degree':9}))
            degree_acc.append(SVM_experiments(data, kernel, {'degree':11}))
            r_time = (time.time() - s_time)/5
            
            acc_list.append(max(degree_acc))
            time_list.append(r_time)
    
    df_results[kernel] = acc_list
    df_time[kernel] = time_list
```

아래의 표를 통해서 각 kernel별 성능을 볼 수 있다.

|    | Dataset                        |   linear |   poly |   sigmoid |   rbf |
|:--:|:------------------------------:|:--------:|:------:|:---------:|:-----:|
|  0 | iris                           |    97.6  |  97.2  |     27.4  | 96.6  |
|  1 | breast_cancer                  |    95.37 |  91.6  |     46.7  | 91.76 |
|  2 | wine                           |    94.92 |  67.8  |     26.78 | 68.81 |
|  3 | penguins                       |    99.45 |  99.55 |     99.73 | 99.45 |
|  4 | titanic                        |   100    |  98.03 |     97.7  | 94.59 |
|  5 | algerian_forest_fires          |    95.93 |  84.2  |     93.33 | 93.21 |
|  6 | breast_cancer_coimbra          |    72.82 |  66.41 |     65.13 | 72.05 |
|  7 | heart_failure_clinical_records |    82.22 |  77.58 |     83.54 | 81.01 |
|  8 | seeds                          |    94.71 |  87.43 |     94.29 | 94.57 |
|  9 | HCV                            |    94.51 |  92.97 |     92.36 | 93.13 |
| 10 | breast_tissue                  |    65.14 |  39.71 |     50.86 | 58    |
| 11 | digits                         |    97.21 |  98.57 |     90.29 | 98.62 |
| 12 | MNIST                          |    87.98 |  86.44 |     85.11 | 91.62 |
| 13 | FashionMNIST                   |    80.59 |  74.48 |     43.37 | 81.11 |
| 14 | KMNIST                         |    75.66 |  78.42 |     69.64 | 83.72 |

좀 더 쉽게 kernel별 성능을 확인하기 위해서 평균 rank를 계산한다.

```py
print(df_results.rank(method='average', axis=1, ascending=False).mean())
```

위 코드를 통해 확인한 kernel별 평균 rank는 아래와 같다.


|   linear |   polynomial |   sigmoid |   rbf |
|:--------:|:------:|:---------:|:-----:|
|    1.63  |  2.93  |     3.27  | 2.17  |

---

### 결과 분석

실험 결과, linear kernel을 사용하는 것이 가장 좋은 성능을 보이며 rbf kernel, polynomial kernel, sigmoid kernel 순으로 성능이 좋았다.

이 실험 결과는 우리가 일반적으로 알고 있는 '보통 비선형 결정경계를 위해 kernel을 활용하면 더 성능이 좋다.'라는 사실과 달랐기 때문에 그에 대한 원인 분석을 해보았다.


|    |   linear |   poly |   sigmoid |   rbf |
|:--:|:--------:|:------:|:---------:|:-----:|
|  0 |      1   |      2 |         4 |   3   |
|  1 |      1   |      3 |         4 |   2   |
|  2 |      1   |      3 |         4 |   2   |
|  3 |      3.5 |      2 |         1 |   3.5 |
|  4 |      1   |      2 |         3 |   4   |
|  5 |      1   |      4 |         2 |   3   |
|  6 |      1   |      3 |         4 |   2   |
|  7 |      2   |      4 |         1 |   3   |
|  8 |      1   |      4 |         3 |   2   |
|  9 |      1   |      3 |         4 |   2   |
| 10 |      1   |      4 |         3 |   2   |
| 11 |      3   |      2 |         4 |   1   |
| 12 |      2   |      3 |         4 |   1   |
| 13 |      2   |      3 |         4 |   1   |
| 14 |      3   |      2 |         4 |   1   |

먼저 이 위의 표는 데이터셋별 rank 전체를 보여준다.

위 표를 자세히 살펴보면 linear kernel의 성능이 좋은 데이터셋은 인덱스 기준 0-10번이며 11-14번 데이터셋은 rbf kernel이 가장 좋았다.

여기서 0-10번 데이터는 정형 데이터이며 데이터 수가 100-600개로 매우 적으며 입력 변수의 개수 역시 4-30개로 적다는 것을 알 수 있다.

반면에 11-14번 데이터는 비정형 이미지 데이터이며 데이터 수와 변수의 개수가 훨씬 큰 것을 볼 수 있다.

즉, 데이터의 수와 변수가 적으면 kernel을 활용하는 것이 성능에 악영향을 준다는 것이다.

#### 과적합 

그 원인을 생각해보면 가장 먼저 떠올릴 수 있는 것은 과적합 문제일 것이다.

과적합은 학습 데이터가 적은 상황에서 모델이 학습 데이터에 너무 fitting되서 일반화 성능이 떨어지는 현상을 말한다.

실제로 과적합 문제가 발생했는지 확인하기 위해서 학습 데이터 성능과 테스트 데이터 성능의 차이를 확인해보았다.

SVM_experiments 함수를 약간 수정하여 학습 데이터에 대한 정확도까지 출력하는 overfitting_check함수를 만들고, linear kernel과 rbf kernel에 대해 실험을 진행하였다.

```py
def overfitting_check(data, kernel, svm_configs={'C':1}):
    seed_list = [0,1,2,3,4,5,6,7,8,9]
    train_acc_list = []
    test_acc_list = []

    for seed in seed_list:
        set_seed(seed)
        
        X_train, X_test, y_train, y_test = train_test_split(data['x'], data['y'], 
                                                            test_size=0.33,
                                                            random_state=seed)
        
        svm = SVC(kernel=kernel, random_state=seed, **svm_configs)
        svm.fit(X_train, y_train)
        
        train_pred = svm.predict(X_train)
        train_acc_list.append(accuracy_score(y_train, train_pred)*100)
        
        test_pred = svm.predict(X_test)
        test_acc_list.append(accuracy_score(y_test, test_pred)*100)
    
    return round(np.mean(train_acc_list),2), round(np.mean(test_acc_list),2)

df_results = pd.DataFrame()

df_results['Dataset'] = data_list
df_time['Dataset'] = data_list

for kernel in ['linear','rbf']:
    train_acc_list = []
    test_acc_list = []

    for key, data in datasets.items():
        
        train_acc, test_acc = overfitting_check(data, kernel)
        
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

    
    df_results[f'{kernel} train acc'] = train_acc_list
    df_results[f'{kernel} test acc'] = test_acc_list
```

|    | Dataset                        |   linear train acc |   linear test acc |   rbf train acc |   rbf test acc |
|:--:|:------------------------------:|:------------------:|:-----------------:|:---------------:|:--------------:|
|  0 | iris                           |              98.3  |             97.6  |           97    |          96.6  |
|  1 | breast_cancer                  |              96.54 |             95.37 |           91.21 |          91.76 |
|  2 | wine                           |              99.75 |             94.92 |           68.57 |          68.81 |
|  3 | penguins                       |              99.82 |             99.45 |           99.6  |          99.45 |
|  4 | titanic                        |             100    |            100    |           99.75 |          94.59 |
|  5 | algerian_forest_fires          |              98.1  |             95.93 |           96.13 |          93.21 |
|  6 | breast_cancer_coimbra          |              76.62 |             72.82 |           88.96 |          72.05 |
|  7 | heart_failure_clinical_records |              85.05 |             82.22 |           90.4  |          81.01 |
|  8 | seeds                          |              95.21 |             94.71 |           93.86 |          94.57 |
|  9 | HCV                            |              98.68 |             94.51 |           97.49 |          93.13 |
| 10 | breast_tissue                  |              76.34 |             65.14 |           68.73 |          58    |
| 11 | digits                         |             100    |             97.21 |           99.56 |          98.62 |
| 12 | MNIST                          |             100    |             87.98 |           98.41 |          91.62 |
| 13 | FashionMNIST                   |             100    |             80.59 |           88.38 |          81.11 |
| 14 | KMNIST                         |             100    |             75.66 |           98.18 |          83.72 |

위 테이블에서 비교한 결과를 보면 breast_cancer_coimbra나 heart_failure_clinical_records 데이터셋에서는 과적합이 발생하여 테스트 성능이 떨어지는 것을 확인할 수 있다.

하지만 다른 대부분의 데이터에서는 학습 데이터에 대한 성능 역시 linear kernel이 우수한 성능을 보였다.

즉, 학습 데이터 부족에 의한 과적합 문제로 인해서 rbf kernel과 같은 비선형 변환 kernel의 성능이 낮은 것이 아니다.

#### 결정경계 시각화

그렇다면 입력 변수의 개수가 적을수록 비선형 변환 kernel의 성능에 악영향을 준다는 것일까?

직관적으로 생각해보면 입력 변수가 많아질수록 데이터는 복잡해지고, 복잡한 구조를 가질수록 비선형 변환을 하는 것이 유리할 것이다.

반대로 단순한 데이터일수록 선형 분리하는 것이 더 쉬워질 것이다.

하지만 데이터의 복잡도를 직접 정량화하는 방법을 모르기 때문에 결정경계를 시각화하여 정성적으로 확인을 해보았다.



---

### 결론

이번 실험을 통해서 어떤 kernel이 일반적으로 좋은 성능을 낼 수 있는지 확인해보았다.

실험 결과 및 분석을 통해서 데이터의 복잡도가 낮을 때에는 linear kernel을 사용하는 것이 좋고

데이터의 복잡도가 높을 때에는 rbf kernel을 사용하는 것이 좋다는 것을 확인할 수 있었다.

하지만 데이터의 수가 적을 때에는 몇 초 이내로 실험결과가 나오기 때문에 모든 kernel을 다 사용해보는 것을 추천한다.


**데이터별 10회 반복실험 요구 시간(초단위)**

|    | Dataset                        |    linear |      poly |   sigmoid |       rbf |
|:--:|:------------------------------:|:---------:|:---------:|:---------:|:---------:|
|  0 | iris                           | 0.0129998 | 0.326424  | 0.0190001 | 0.0159991 |
|  1 | breast_cancer                  | 3.53016   | 0.0225989 | 0.0489995 | 0.0340052 |
|  2 | wine                           | 0.295108  | 0.0168794 | 0.0200005 | 0.0169954 |
|  3 | penguins                       | 0.0160055 | 0.0197972 | 0.0139999 | 0.0159717 |
|  4 | titanic                        | 0.0130272 | 0.0160017 | 0.0140002 | 0.0159998 |
|  5 | algerian_forest_fires          | 0.0129726 | 0.0150005 | 0.0139997 | 0.0150001 |
|  6 | breast_cancer_coimbra          | 0.0140071 | 0.0139982 | 0.013     | 0.0129998 |
|  7 | heart_failure_clinical_records | 0.0209594 | 0.0221982 | 0.0250003 | 0.0210276 |
|  8 | seeds                          | 0.014045  | 0.016208  | 0.0129998 | 0.0139725 |
|  9 | HCV                            | 0.0269878 | 0.0318243 | 0.0249996 | 0.0349994 |
| 10 | breast_tissue                  | 0.014014  | 0.0133984 | 0.013     | 0.0140002 |
| 11 | digits                         | 0.141985  | 0.157818  | 0.444999  | 0.348028  |
| 12 | MNIST                          | 0.844005  | 1.45006   | 1.06814   | 1.69422   |
| 13 | FashionMNIST                   | 0.827233  | 0.957336  | 1.21803   | 1.53414   |
| 14 | KMNIST                         | 1.15817   | 1.8599    | 1.26241   | 2.3484    |
