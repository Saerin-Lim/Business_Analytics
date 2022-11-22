
## Bagging & Boosting in terms of base learner 튜토리얼

#### > [Bagging & Boosting](https://github.com/Saerin-Lim/Business_Analytics/blob/master/4.ensemble%20learning/Bagging%20and%20Boosting%20slide.pdf) 설명 보러가기 <

앙상블 기법은 크게 bagging 기반 알고리즘과 boosting 기반 알고리즘으로 분류할 수 있다.

Bagging 기반 알고리즘에는 대표적으로 random forest가 있고 Boosting 기반 알고리즘에는 Adaboost나 GBM이 있다.

이 둘은 모델을 병렬적으로 학습을 시키는가 아니면 순차적으로 학습을 시키는가에 따라 분류할 수 있다.

Bagging 기반 알고리즘은 부스트랩을 활용해 모델을 병렬적으로 학습하여 다양한 데이터 분포를 학습할 수 있기 때문에 예측값의 분산을 줄일 수 있다. 

따라서 일반적으로 편향이 크고 분산이 큰 복잡한 모델을 활용하며 대표적인 Bagging 기반 알고리즘인 random forest 역시 decision tree를 base learner로 활용한다.

반면에 Boosting 기반 알고리즘은 대부분 stemp tree와 같은 weak learner를 base learner로 활용하는데, 이 weak learner는 예측값에 대한 분산이 낮고 편향이 높다.

정리하자면 Bagging 기반 알고리즘은 복잡도가 높은 모델을 base learner로 사용하고 Boosting 기반 알고리즘은 복잡도가 낮은 모델을 base learner로 사용한다.

이번 튜토리얼에서는 base learner를 바꿔보면서 bagging과 boosting 기반 알고리즘들의 성능과 학습 시간을 비교하고 왜 이러한 경향성을 가지게 되었는지 분석하는 것을 목적으로 한다.

![image](https://user-images.githubusercontent.com/80674834/203309354-81cd17ca-7e9d-4759-9c1c-0b946285e115.png)

이미지 출처 : https://towardsdatascience.com/ensemble-learning-bagging-boosting-3098079e5422
