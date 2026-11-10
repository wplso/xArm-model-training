# 무부하 No.3·4·5 초기 학습 모델

2026년 10월 31일 Hiwonder xArm의 빈 그리퍼 Pick-and-Place 궤적에서 학습한 모델입니다. 학습은 정상 회차만 사용했습니다. 점수와 경보는 한 회차가 끝난 뒤 계산하며, 평가는 같은 날짜에 수집한 정상 회차의 오탐률을 비교합니다. 41g 운반 평가, 이상 Recall, 다른 날짜 재현성, Raspberry Pi 4 실행 시간은 별도 검증 범위입니다.

| 폴더 | 보존한 모델 | 정상 학습 / 임계값 교정 / 평가 |
|---|---|---|
| `classical_no3` | 24개 joblib. 두 센서 구성과 Statistical·Isolation Forest·PCA를 정상 학습 수 50·100·200·250에서 비교 | 최종 학습 250 / 100 / 150 |
| `classical_multi_action` | 18개 joblib. 두 센서 구성과 세 알고리즘을 동작별·통합·동작 조건부 정규화로 비교. 동작별 묶음에는 세 동작의 개별 모델이 들어 있음 | No.3 250 / 100 / 150, No.4와 No.5 각 10 / 10 / 10 |
| `autoencoders` | PyTorch 가중치 36개와 모델별 정규화·임계값. 요약 특징 MLP, 시계열 MLP, 1D-CNN을 두 센서 구성·두 학습 범위·세 초기값으로 비교. 사전 지정한 첫 초기값의 ONNX 12개 포함 | No.3 전용 250 / 100 / 150, 통합 270 / 120 / 170 |

## 대표 결과

| 모델과 입력 | 정상 평가 | 오탐률 |
|---|---:|---:|
| No.3 PCA · 전류·전압 / 진동·위치 추가 | 150 | **8/150 (5.33%) / 31/150 (20.67%)** |
| No.3 1D-CNN AE · 전류·전압 / 진동·위치 추가 | 초기값별 150 | 평균 **5.11% / 9.56%** |
| No.3·4·5 통합 Isolation Forest · 진동·위치 포함 | 170 | **1/170 (0.59%)**, 동작별 0/150·1/10·0/10 |
| No.3·4·5 통합 1D-CNN AE · 진동·위치 포함 | 초기값별 170 | 평균 가중 **3.33%**, 동작별 동등 가중 **15.78%** |

No.4·5의 평가 수는 각각 10회입니다. 통합 가중 오탐률에는 평가 회차가 많은 No.3이 크게 반영됩니다. Autoencoder 점수는 정규화된 입력 중 유효한 원소의 평균 제곱 재구성 오차입니다. 세 계열 모두 정상 교정 점수의 95분위수(`higher`)를 사용하고 점수가 임계값보다 클 때 경보를 냅니다.

## 저장 모델 점수 계산

명령은 저장소 루트에서 실행합니다. 원래 파일을 다시 열어 확인한 환경은 Windows CPU Python 3.13.15입니다. 고전 모델은 numpy 2.5.4, scipy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1, joblib 1.6.0을 사용했습니다. Autoencoder는 PyTorch 2.15.0+cpu와 numpy 2.5.4, ONNX 비교에는 onnx 1.23.2와 onnxruntime 1.31.1을 사용했습니다. 신뢰하는 출처의 joblib만 로드하세요.

```powershell
python models/unloaded/score_saved.py classic models/unloaded/classical_no3/models/external_PCA_train250.joblib features.csv
python models/unloaded/score_saved.py ae models/unloaded/autoencoders/models/No3_external_sequence_CNN1D_AE_seed20261031 inputs.npy
```

- `features.csv`는 한 행에 종료된 회차 하나를 담습니다. 각 `experiment_freeze.json`의 `config.feature_sets`에 입력 컬럼명이 있으며, No.3·4·5 묶음에는 동작을 선택할 `action_group_id`가 필요합니다.
- `inputs.npy`는 모델별 `model.json`의 `shape` 앞에 회차 차원을 붙인 정규화 전 배열입니다. 요약 특징은 `(회차, 6 또는 23)`이며 `classical_no3/experiment_freeze.json`과 같은 입력 순서를 씁니다. 시계열은 `(회차, 2 또는 11, 64)`이며 `model.json`의 채널 순서를 씁니다. 64개 구간은 명령 시작부터 기존 완료 판정까지 등간격으로 나눕니다. 전류·전압은 구간 평균과 과거값 유지 상한 0.35초, 진동은 3표본 이상인 구간의 축별 표준편차, 관절은 구간 중심 직전 유효 위치를 최대 0.75초 유지해 구성합니다. 관측이 없는 값은 `NaN`으로 표시합니다.
- `model.onnx`는 표준화된 입력에서 복원 결과를 반환합니다. `normalizer.npz`와 결측 마스크로 같은 전처리와 점수를 적용하고 `model.json`의 임계값으로 판정합니다.
- `VERIFICATION.json`에는 가중치 전용 로드와 원래 점수·경보 재현 결과가 있습니다. 42개 고전 모델 묶음, 36개 PyTorch 모델, 실제 ONNX 12개를 로컬 평가 자료로 검사했습니다. 평가 CSV에는 회차별 원본 대신 집계값을 담았습니다.

Autoencoder는 모델당 362~12,052개 파라미터이며 기록된 Windows CPU 단일 스레드에서 학습 1.25~8.23초, 회차 1개 PyTorch 추론 중앙값 0.028~0.106ms였습니다. 이 시간은 입력 특징 계산과 파일 읽기를 제외한 모델 실행 시간입니다.
