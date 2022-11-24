
## Bagging & Boosting in terms of base learner 튜토리얼

#### > [Bagging & Boosting](https://github.com/Saerin-Lim/Business_Analytics/blob/master/4.ensemble%20learning/Bagging%20and%20Boosting%20slide.pdf) 설명 보러가기 <

앙상블 기법은 크게 bagging 기반 알고리즘과 boosting 기반 알고리즘으로 분류할 수 있다. 이 둘은 모델을 병렬적으로 학습을 시키는가 아니면 순차적으로 학습을 시키는가에 따라 분류할 수 있다.

Bagging 기반 알고리즘은 부스트랩을 활용해 모델을 병렬적으로 학습하여 다양한 데이터 분포를 학습할 수 있기 때문에 예측값의 분산을 줄일 수 있다. 따라서 일반적으로 편향이 크고 분산이 큰 복잡한 모델을 활용하며 대표적인 Bagging 기반 알고리즘인 random forest 역시 decision tree를 base learner로 활용한다. 

반면에 Boosting 기반 알고리즘은 대부분 stemp tree와 같은 weak learner를 base learner로 활용하는데, 이 weak learner는 예측값에 대한 분산이 낮고 편향이 높다. 

정리하자면 Bagging 기반 알고리즘은 복잡도가 높은 모델을 base learner로 사용하고 Boosting 기반 알고리즘은 복잡도가 낮은 모델을 base learner로 사용한다. 

이번 튜토리얼에서는 base learner를 바꿔보면서 bagging과 boosting 기반 알고리즘들의 성능과 학습 시간을 비교하고 이론과 같은 결과가 나오는지 확인하는 것을 목표로 한다.

추가적으로 Random forest에서는 앙상블의 다양성을 확보하기 위해서 각 tree에서 전체 변수 중 부분 집합을 분기마다 뽑아서 활용한다.

즉, random forest와 bagging에서 base learner를 decision tree로 활용하는 것은 분기변수를 어떤 집합에서 뽑느냐에서 차이를 가진다.

이번 튜토리얼에서는 분기변수를 무작위로 뽑는 것이 앙상블 성능에 어떠한 영향을 주는지까지 확인해 본다.

![image](https://user-images.githubusercontent.com/80674834/203309354-81cd17ca-7e9d-4759-9c1c-0b946285e115.png)

이미지 출처 : https://towardsdatascience.com/ensemble-learning-bagging-boosting-3098079e5422

---

### 데이터 불러오기 및 전처리

이번 튜토리얼에서 활용할 데이터셋은 MNIST로 아래의 그림처럼 0~9까지의 숫자가 28by28 gray scale을 가진 이미지로 표현되어 있으며, 총 70000장의 이미지로 이루어져 있다.

MNIST 데이터셋은 앙상블 기반 방법론을 학습하기에 충분한 관측치와 적당한 복잡도(feature의 개수)를 가지고 있어 해당 데이터셋을 활용하였다.

![image](https://user-images.githubusercontent.com/80674834/203486787-2983dedc-8ad7-4bc4-b1ac-a5a1b5aaaca5.png)

본 튜토리얼에서는 MNIST에서 기본적으로 분할되어 있는 학습 데이터 60000장, 테스트 데이터 10000장을 그대로 사용한다.

그리고 0~255로 표현된 RGB값을 0~1로 스케일링 해주었으며, 앙상블 알고리즘에 입력하기 위해서 (28,28)로 표현된 이미지를 784차원의 1d vector로 reshape 해준다.

```py
import torchvision.datasets as datasets

import warnings
warnings.filterwarnings(action='ignore')

trainset = datasets.MNIST(root='./data', train=True, download=False)
testset = datasets.MNIST(root='./data', train=False, download=False)

X_train, y_train = trainset.data.numpy().reshape(-1,28*28), trainset.targets.numpy()
X_test, y_test = testset.data.numpy().reshape(-1,28*28), testset.targets.numpy()

X = {'train':X_train, 'test':X_test}
y = {'train':y_train, 'test':y_test}

print(f'X train : {X_train.shape}, y train : {y_train.shape}')
print(f'X test : {X_test.shape}, y test : {y_test.shape}')
```

---

### 실험 설계

decision tree는 max depth를 줄이거나 늘림으로써 모델의 복잡도를 쉽게 변경할 수 있다. 예를 들어 max depth를 1로 설정한다면 stamp tree가 되어 일반적으로 boosting에서 base learner로 활용하는 weak learner가 된다. 

따라서 이번 튜토리얼에서 base learner를 decision tree로 사용하고 max depth를 변경해 가면서 base learner의 복잡도를 조정한다.

본 튜토리얼에서는 base learner의 복잡도에 따라서 bagging과 boosting 알고리즘의 성능 및 학습 시간이 어떻게 변화하는지 확인하기 위해서 아래와 같은 실험을 진행한다.

1. Bagging에서 base learner를 decision tree로 설정하고 max depth를 [1,2,4,8,16]으로 변화하며 실험

2. Boosting알고리즘 중 Adaboost에서 base learner를 decision tree로 설정하고 max depth를 [1,5,10,25,50]으로 변화하며 실험

또한, random forest와 decision tree기반 bagging 알고리즘의 차이를 확인하기 위해서 아래와 같은 실험을 추가적으로 진행한다.

3. max depth를 50으로 고정하고 random forest와 decision tree기반 bagging알고리즘 실험

각 실험에서 시드에 따른 변동성을 고려해 5회 반복 실험을 하며 앙상블 모델의 성능은 정확도를 통해 평가하며, 앙상블 개수는 50개로 고정한다.(sklearn adaboost default값)

아래 코드는 실험을 위한 함수들이다. 먼저 ensemble_exp 함수는 특정 max_depth와 seed를 입력받아서 method에 맞는 앙상블 모델을 학습하고 정확도를 반환한다.

그리고 repeat_exp 함수는 원하는 max_depth와 seed만큼 반복실험을 진행하고 각 max_depth와 seed 별 앙상블 모델의 정확도와 학습 시간을 컬럼으로 있는 results df를 반환한다.

```py
import time
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from utils import set_seed

def ensemble_exp(X:dict, y:dict, method:str='Bagging', max_depth:int=1, seed:int=0):
    
    # set seed
    set_seed(seed)
    
    # set base learner & ensemble model
    base_learner = DecisionTreeClassifier(max_depth=max_depth)
    
    if method == 'Bagging':
        ensemble = BaggingClassifier(base_estimator=base_learner,
                                     n_estimators=50,
                                     random_state=seed)
        
    elif method == 'Boosting':
        ensemble = AdaBoostClassifier(base_estimator=base_learner,
                                      n_estimators=50,
                                      random_state=seed)
    else:
        ensemble = RandomForestClassifier(base_estimator=base_learner,
                                          n_estimators=50,
                                          random_state=seed)
    
    # train ensemble model
    ensemble.fit(X['train'], y['train'])
    
    # evaluate ensemble model with testset
    y_pred = ensemble.predict(X['test'])
    accuracy = accuracy_score(y['test'], y_pred)
    
    return accuracy
    
def repeat_exp(X, y, method, depth_list, seed_list):
    
    # experiments
    print(f'{method} based Ensemble training start...')
    results_df = pd.DataFrame(columns=['method','seed','max_depth','acc','time'])
    for depth in depth_list:
        for seed in seed_list:
            # check training time
            since = time.time()
            acc = ensemble_exp(X, y, method, depth, seed)
            after = time.time()-since
            
            crt_results = {'method':method,
                           'seed':seed,
                           'max_depth':depth,
                           'acc':round(acc*100,2),
                           'time':round(after,2)}
            crt_df = pd.DataFrame([crt_results])
            results_df = pd.concat([results_df, crt_df],ignore_index=True)
            
            print(f'method : {method} | seed : {seed} | max_depth : {depth}')
            print(f'acc : {round(acc*100,2)} | time : {round(after,2)} \n')
    
    return results_df
```

---

### 실험

위에서 작성한 함수를 바탕으로 아래의 조건에서 실험을 한다.

1. seed list = [0,1,2,3,4] -> 5회 반복실험

2. max_depth list = [1,2,4,8,16] -> 모델 복잡도가 점점 증가

#### Bagging 실험

위 조건을 바탕으로 먼저 bagging 실험을 진행한다.

```py
# set seed list
seed_list = [0,1,2,3,4]

# set max depth list
depth_list = [1,5,10,25,50]

# Decision tree based bagging experiments
bagging_df = repeat_exp(X, y, 
                        method='Bagging', 
                        seed_list=seed_list,
                        depth_list=depth_list)
```
Bagging based Ensemble training start...

method : Bagging | seed : 0 | max_depth : 1

acc : 37.2 | time : 23.61 

method : Bagging | seed : 1 | max_depth : 1

acc : 32.66 | time : 23.34 


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ㆍ

method : Bagging | seed : 3 | max_depth : 16

acc : 95.59 | time : 254.77 

method : Bagging | seed : 4 | max_depth : 16

acc : 95.46 | time : 253.62 

#### Boosting 실험

다음으로 boosting 실험을 진행한다.

```py
# Adaboost(boosting method) experiments
boosting_df = repeat_exp(X, y, 
                        method='Boosting', 
                        seed_list=seed_list,
                        depth_list=depth_list)
```

Boosting based Ensemble training start...

method : Boosting | seed : 0 | max_depth : 1

acc : 72.99 | time : 46.48 

method : Boosting | seed : 1 | max_depth : 1

acc : 72.99 | time : 46.51 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ㆍ

method : Boosting | seed : 3 | max_depth : 16

acc : 96.41 | time : 427.12 

method : Boosting | seed : 4 | max_depth : 16

acc : 96.29 | time : 428.65 

#### Random forest 실험

마지막으로 decision tree based bagging과 random forest의 차이를 확인하기 위해서 random forest실험을 진행한다.

```py
# Random forest experiments
rf_df = repeat_exp(X, y, 
                    method='Random_forest', 
                    seed_list=seed_list,
                    depth_list=depth_list)
```
Random_forest based Ensemble training start...

method : Random_forest | seed : 0 | max_depth : 1

acc : 53.71 | time : 1.27 

method : Random_forest | seed : 1 | max_depth : 1

acc : 51.06 | time : 1.25 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ㆍ

method : Random_forest | seed : 3 | max_depth : 16

acc : 96.57 | time : 17.03 

method : Random_forest | seed : 4 | max_depth : 16

acc : 96.36 | time : 17.02 

---

### 실험 결과

실험 결과를 직관적으로 확인하기 위해 그래프를 통해 결과를 시각화하고 분석한다.

* Bagging vs Boosting

#### in terms of accuracy

```py
import seaborn as sns
import matplotlib.pyplot as plt

# accuracy vs max_depth line plot
plt.figure(figsize=(10,8))

sns.lineplot(data=bagging_df, x='max_depth', y='acc', label='Bagging')
sns.lineplot(data=boosting_df, x='max_depth', y='acc', label='Boosting')

plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.title('Bagging vs Boosting in terms of accuracy')

plt.legend()
plt.show()

# time vs max_depth line plot
plt.figure(figsize=(10,8))

sns.lineplot(data=bagging_df, x='max_depth', y='time', label='Bagging')
sns.lineplot(data=boosting_df, x='max_depth', y='time', label='Boosting')

plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.title('Bagging vs Boosting in terms of training time')

plt.legend()
plt.show()
```

![image](https://user-images.githubusercontent.com/80674834/203702855-732e814d-5d1b-4730-86d5-fac1ce9f0386.png)

위 그래프들은 max depth를 x축, accuracy와 training time을 각각 y축으로 하여 모델의 복잡도에 따른 성능과 학습 시간 변화를 볼 수 있다. 선 주변에 연한 밴드는 5회 반복실험에서 95% 신뢰 구간을 나타낸다.

먼저 왼쪽 그래프를 보면 모델 성능과 max_depth의 관계를 보면 bagging기반 앙상블과 boosting기반 앙상블 모두 성능이 증가하는 것을 볼 수 있다. 모델 복잡도가 증가할수록 모델의 성능이 증가하는 것은 일반적인 현상이다.

하지만 우리가 주목해야할 부분은 모델 복잡도가 극단적으로 낮은 max_depth=1인 경우, 즉 stamp tree가 base learner인 경우이다.

Bagging에서 base learner가 stamp tree인 경우 성능이 약 34%정도로 매우 저조한 것을 확인할 수 있다. 반면에 boosting 기반 앙상블인 adaboost의 경우에는 성능이 약 73%로 꽤 준수한 성능을 보인다.

Bagging기반 앙상블은 모델을 병렬적으로 학습시키고 aggrigation하기 때문에 각 모델의 성능이 어느정도 뛰어나야 하지만, boosting기반 앙상블은 모델을 순차적으로 학습시키면서 이전 모델의 단점을 보완하는 식으로 학습하기 때문에 weak learner로도 충분히 잘 학습된다는 사실을 실험적으로 볼 수 있다.

#### in terms of training time

오른쪽 그래프를 통해서 base learner의 모델 복잡도와 앙상블 학습 시간에 대한 관계를 볼 수 있다. 기본적으로 두 방법 모두 base learner의 모델 복잡도가 증가할수록 학습 시간이 길어지는 것을 확인할 수 있다. 

또한 병렬적으로 base learner를 학습하는 bagging 기반 앙상블에 비해서 순차적으로 모델을 학습하는 boosting 기반 앙상블이 학습 시간을 더 많이 요구하며, 모델 복잡도가 증가할수록 학습 시간이 증가하는 폭 역시 훨씬 큰 것을 실험적으로 확인 할 수 있다.

또 하나 흥미로운 점은 boosting 기반 앙상블에서 base learner를 조금만 더 좋은 모델로 활용하면 성능이 큰 폭으로 상승한다는 것이다.

일반적으로 boosting 기반 앙상블은 모델의 복잡도를 높이는 것보다는 base learner의 개수를 늘리는 것이 시간 측면에서 훨씬 효율적이라고 알려져 있기 때문에 이 둘을 비교하는 실험도 흥미로운 주제가 될 수 있을 것 같다.

---

* decision tree based bagging vs random forest

```py
# decision tree based bagging vs random forest in terms of accuracy
plt.figure(figsize=(10,8))

sns.lineplot(data=bagging_df, x='max_depth', y='acc', label='Bagging')
sns.lineplot(data=rf_df, x='max_depth', y='acc', label='random forest')

plt.xlabel('Max depth')
plt.ylabel('Accuracy')
plt.title('Decision tree based bagging vs Random Forest in terms of accuracy')

plt.legend()
plt.show()

# decision tree based bagging vs random forest in terms of training time
plt.figure(figsize=(10,8))

sns.lineplot(data=bagging_df, x='max_depth', y='time', label='Bagging')
sns.lineplot(data=rf_df, x='max_depth', y='time', label='random forest')

plt.xlabel('Max depth')
plt.ylabel('Training time')
plt.title('Decision tree based bagging vs Random Forest in terms of training time')

plt.legend()
plt.show()
```
![image](https://user-images.githubusercontent.com/80674834/203703309-57eb3bca-d1a1-4ff4-97e7-3f0538adbcaa.png)

이번 튜토리얼에서 개인적으로 가장 궁금했던 부분이었다.

왼쪽 그래프를 보면 당연히 모든 부분에서 random forest가 좋은 성능을 보이고 있다. 이 실험 결과는 분기변수를 모두 사용하는 것이 아니라 랜덤하게 부분집합을 정의해서 사용하는 것이 앙상블 성능 향상에 긍정적인 영향을 주고 있다는 것을 보여준다. 그리고 그 영향이 예상보다 상당히 크다는 것을 볼 수 있다.

이 실험을 통해서 앙상블을 형성할 때, 다양성을 확보하는 것이 얼마나 중요한지를 간접적으로 확인할 수 있다.

또한, 학습 시간 측면에서도 random forest가 훨씬 빠른 것을 볼 수 있었다. 일반적인 decision tree는 모든 분기변수에 대한 전수조사를 하는 것에 비해서 random forest는 몇몇 변수만을 선정하기 때문에 이런 차이가 발생했다고 볼 수 있다.

---

### 결론

이번 튜토리얼에서는 bagging 기반 앙상블과 boosting 기반 앙상블의 차이를 base learner의 복잡도 관점에서 확인해 보았다.

실험 결과를 통해서 'bagging에는 모델 복잡도가 큰 learner를 사용하고 boosting에는 모델 복잡도가 작은 learner를 주로 사용한다.'고 수업에서 배웠던 내용을 실험적으로 확인할 수 있었으며, 모델 복잡도에 따른 성능 향상과 학습 시간 사이의 trade-off를 볼 수 있었다. (특히, boosting 계열 알고리즘에서 왜 그렇게 학습 시간을 줄이려는 노력을 했는지 직접 실험을 돌려보면서 느낄 수 있었다...)

또한, 단순히 decision tree based bagging과 random forest를 비교하는 실험을 통해서 앙상블의 다양성을 확보하는 것이 얼마나 중요한 것인지 직접적으로 확인할 수 있었다. 이 둘의 차이는 성능 뿐만 아니라 학습 시간 관점에서도 매우 컸는데 '일단 random forest 돌려봐'라는 격언을 생각나게 하는 결과였다. 
