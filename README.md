# NN Geometry Power Analysis

저전력 엣지 컴퓨팅 환경에서 **CNN 계층 구조의 기하학적 형상**(깊이, 각 계층의 채널 폭 배치, 풀링 위치)이 추론 정확도와 에너지 효율에 어떤 영향을 주는지 분석하는 연구입니다.

같은 모델 규모라도 채널을 어떻게 배치하느냐에 따라 CPU 추론 시간과 소비 에너지가 달라질 수 있다는 가설을 검증합니다. CIFAR-10 분류용 CNN을 학습하고, ONNX 및 CodeCarbon 기반의 CPU 측정으로 구조별 결과를 기록·비교합니다.

## 진행 현황

- 깊고 좁은형, 얕고 넓은형, 균일형, 모래시계형 등 6개 CNN을 CIFAR-10에서 50 epoch 학습했습니다.
- 기존 6개 모델의 최종 정확도는 약 **81.37%~84.36%**입니다.
- CPU에서 전체 모델과 개별 계층의 추론 시간·에너지를 측정하고, 그래프로 비교하는 환경을 구성했습니다.
- 현재는 매개변수 수, 깊이, 풀링 위치를 통제한 다양한 구조를 자동 생성하여 정확도-에너지 관계를 더 넓게 분석하는 단계입니다.

## 파일 구조

```text
NN-Geometry-Power-Analysis/
├── FlexibleCNN.py             # CNN 구조 생성, 채널 형상·풀링 위치 설정
├── generate_dataset.py        # 무작위 CNN을 생성하고 ONNX 및 구조 메타데이터로 저장
├── load_data.py               # CIFAR-10 다운로드·변환 및 PyTorch DataLoader 제공
├── RunCNN.py                  # 6개 기준 CNN의 CPU 추론 에너지 측정
├── run_onnx.py                # ONNX Runtime의 단일 모델 추론 지연 시간 측정
├── verify_onnx.py             # PyTorch와 ONNX 모델의 정확도·출력 일치 여부 확인
├── Analyze.py                 # 에너지 로그를 정규화하고 비교 그래프 생성
├── dataset_structure.csv      # 자동 생성 모델의 구조·매개변수·ONNX 메타데이터
├── nn_geometry_power_log.csv  # 모델 및 계층별 시간·에너지 측정 로그
├── emissions.csv              # CodeCarbon이 기록한 배출량·에너지 원본 로그
├── train_result.txt           # 기준 CNN 학습 과정과 정확도 기록
├── model_onnx/                # 생성·내보낸 ONNX 모델 파일
├── data_cifar10/              # 변환된 CIFAR-10 데이터 캐시
├── data_cache/                # CIFAR-10 원본 다운로드·압축 해제 캐시
└── plots/                     # 구조별 에너지 비교 및 계층별 흐름 그래프
```

## 핵심 구성 요소

- `FlexibleCNN.py`: 증가형·감소형·균일형·모래시계형·역모래시계형 채널 폭 패턴을 만들고, 모델별 구조 정보를 `ModelConfig`로 관리합니다.
- `generate_dataset.py`: 구조, 깊이, 채널 수, 풀링 위치, 매개변수 수를 무작위로 조합한 CNN을 ONNX로 내보내고 `dataset_structure.csv`에 재현 가능한 기록을 남깁니다.
- `RunCNN.py`: CodeCarbon을 사용해 CPU에서 반복 추론할 때의 전체 모델 및 개별 계층 에너지 사용량을 측정합니다.
- `Analyze.py`: 측정값을 추론 1회당 에너지(J), 평균 전력(W)으로 변환하고 구조별 비교 그래프를 생성합니다.

## 분석 목표

구조별 정확도, 추론 지연 시간, 추론 1회당 에너지를 함께 비교하여 정확도-에너지 Pareto 전선을 찾고, 자원이 제한된 기기에서도 활용할 수 있는 효율적인 CNN 계층 폭 배치 원칙을 제안하는 것이 목표입니다.

## 주요 사용 도구

Python, PyTorch, torchvision, ONNX/ONNX Runtime, CodeCarbon, pandas, matplotlib, seaborn

## 라이선스

이 프로젝트는 [MIT License](LICENSE)를 따릅니다.
