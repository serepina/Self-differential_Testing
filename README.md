# Self-Differential Testing 

자가 차동 시험을 이용한 분류 모델의 성능 근사

Approximating the Accuracy of Classification Models Using Self-Differential Testing


## 요약
---
차동 시험(differential testing)은 유사한 응용 프로그램이 동일한 입력에 대해 서로 다른 출력을 생성하는지 관찰하여 오류를 감지하는 전통적인 소프트웨어 시험 기법이다. 인공지능 시스템에서도 차동 시험이 사용되고 있는데, 현존하는 연구 방법들은 시험 대상 신경망과 동일 기능을 수행하는 구조가 다른 고품질의 참조 대상 신경망을 찾는 비용을 요구한다. 본 논문에서는 인공진으 시스템의 차동 시험시 다른 구조의 신경망을 찾을 필요 없이 시험 대상 신경망을 이용해 참조 모델을 만들어 시험을 수행하는 자가 차동 시험(self-differential testing) 기법을 제안한다. 실험 결과 제안 기업은 다른 참조 모델을 필요로 하는 기존 방법보다 저비용으로 유사한 효과를 내는 것을 확인하였다. 본 논문은 자가 차동 시험의 응용인 자가 차동 분석을 활용해 분류 신경망의 정확도 근사 방법도 추가로 제안한다. 제안 기법을 통한 근사 정확도는 MNIST와 CIFAR10의 유사 데이터셋을 이용한 실험에서 실제 정확도와 0.0002~0.09 정도의 낮은 차이로 성능 근사의 가능성을 확인할 수 있었다.

## Requirements
---
- numpy
- tensorflow

## Usage
---
MNIST Train: 'python model_mnist/train.py'

MNIST Self-Differential Testing: 'python model_mnist/diff.py --m1 Target_model_name --m2 Reference_model_name --data Dataset_name --method testing'

MNIST Self-Differential Analysis: 'python model_mnist/diff.py --m1 Target_model_name --m2 Reference_model_name --data Dataset_name --method analysis'

CIFAR10 train: 'python model_cifar/train.py'

CIFAR10 Self-Differential Testing: 'python model_cifar/diff.py --m1 Target_model_name --m2 Reference_model_name --data Dataset_name --method testing'

CIFAR10 Self-Differential Analysis: 'python model_cifar/diff.py --m1 Target_model_name --m2 Reference_model_name --data Dataset_name --method analysis'
