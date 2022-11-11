## autoencoder-based anomaly detection 튜토리얼

#### > [autoencoder-based anomaly detection](https://github.com/Saerin-Lim/Business_Analytics/blob/master/3.anomaly%20detection/model-based%20anomaly%20detection%20slide.pdf) 설명 보러가기 <

개인적으로 오토인코더를 사용할 때 가장 애매하고 고민을 많이 하게 되는 부분은 모델구조 부분이다. 특히 layer를 몇 개나 쌓을건지와 hidden vector의 dimension을 어떻게 설정할지가 가장 어려웠다.

이번 튜토리얼에서는 위와 같은 고민을 해결하기 위해서 layer의 개수와 hidden vector dimension에 따라서 어떻게 오토인코더의 복원성능과 이상탐지 성능이 변화하는지 살펴보는 것을 목표로 한다.

### 데이터셋 불러오기

이번 튜토리얼에서 활용할 데이터셋은 대표적인 이미지 데이터인 CIFAR-10을 활용한다. CIFAR-10은 아래 그림처럼 총 10개의 class로 이루어져 있으며 각 class마다 32 by 32 사이즈의 RGB 이미지가 6000개씩 총 60000개의 이미지로 구성되어 있다.

![cifar-10](https://user-images.githubusercontent.com/80674834/201267578-bce70474-3354-4ed1-a081-a39609f134ba.PNG)`

