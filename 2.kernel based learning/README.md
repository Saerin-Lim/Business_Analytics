## SVM 튜토리얼(SVM 설명 보러가기)

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

#### 데이터셋 불러오기

가장 먼저 데이터셋을 불러와 딕셔너리 형태로 저장한다.

데이터를 불러오는 과정에서 간단한 전처리를 진행하고 numpy ndarray형태로 변환한다.

1. 결측치 제거

2. 범주형 변수 one-hot encoding

3. target 변수 분리 및 one-hot encoding

4. scaling(정형 데이터 : standard scaling, 비정형 데이터 : minmax scaling)

데이터셋이 여러 개이기 때문에 모든 데이터를 주피터노트북으로 하기 어려워 utils.py에서 각 데이터마다 불러오기 및 전처리 하는 함수 load_data를 정의하였다.
