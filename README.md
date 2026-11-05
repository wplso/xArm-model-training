# xArm V3 정상 기반 이상 탐지 학습 코드

Hiwonder xArm 실측 데이터로 개발한 **V3** 학습 코드를 공유하는 저장소입니다. Statistical Anomaly Score, Isolation Forest, PCA 재구성 오차를 비교합니다. V3.1은 포함하지 않습니다.

## 먼저 볼 파일

| 파일 | 용도 |
|---|---|
| `train_v3.py` | 특징 CSV에서 V3 모델을 학습하는 독립 실행 진입점 |
| `v3/detectors.py` | 원래 학습에 사용한 모델·전처리·점수·임계값 구현 |
| `build_features.py` | 저장된 원시 시계열과 사이클 목록에서 특징 계산 |
| `config/V3_TRAINING_CONFIG.json` | 11개 입력 구성, seed, 원래 데이터 분할과 설정 |
| `config/FEATURE_DEFINITIONS.json` | 구간 경계·샘플 수·결측 처리 규칙 |
| `original/train_xarm_v3.py` | 실제 V3 개발 때 실행한 원본 학습·평가 코드 |
| `original/plot_xarm_v3.py` | 원본 결과 그래프 코드 |
| `original/normal_models.py` | 원래 사용한 공통 모델 모듈 전체 |
| `docs/SOURCE_PROVENANCE.json` | 원본 파일 위치와 SHA256, 함수 추출 이력 |
| `docs/VALIDATION.json` | 원래 V3 모델과의 재현 검증 결과 |

`original/`은 코드 검토·추적용 원본입니다. 원본 진입점은 기존 전체 프로젝트의 이전 분석 결과·모델·원시 파일 트리가 필요합니다. 이 저장소만으로 실행할 때는 아래의 `train_v3.py`를 사용합니다. 이 진입점은 원본과 같은 `Detector`·특징 순서·정규화·임계값 규칙을 사용합니다. 원본의 이전 버전 모델 비교와 전체 보고서 생성까지 독립 실행기로 옮긴 것은 아닙니다.

원시 데이터·특징 CSV·학습된 모델은 이 공개 저장소에 포함하지 않습니다. 팀 내에서 별도로 공유받은 파일을 사용하세요. `.gitignore`에 데이터와 모델을 제외했습니다.

## 환경과 실행

원래 검증 환경은 Python 3.13.15입니다. 고정 라이브러리 버전은 `requirements.txt`에 있습니다. Python은 `hashlib.file_digest`를 지원하는 3.11 이상이 필요하며, 동일 결과 재현에는 원래 검증 환경을 권장합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

기존 V3의 `features/CYCLE_FEATURES.csv`를 공유받았다면 바로 학습할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe train_v3.py --features "C:\공유데이터\CYCLE_FEATURES.csv" --output results/v3_replay
```

`cycle_id`, `cohort`와 설정에 나열된 특징 컬럼이 필요합니다. 원래 특징 CSV에 포함된 `cohort`를 그대로 사용하세요. 임의 행 무작위 분할로 바꾸지 않습니다.

선택적인 `classification` 컬럼이 있으면 train/validation의 라벨은 `confirmed_normal` 또는 기존 실측 데이터의 `confirmed_normal_41g`여야 합니다. 이상 라벨이 학습·검증에 들어오면 중단합니다.

- `cohort=train` 정상 250회로 결측 중앙값과 StandardScaler 및 모델을 적합합니다.
- `cohort=validation` 정상 100회 점수의 95% 분위수(`method="higher"`)로 임계값을 결정합니다.
- `holdout` 정상 100회, 별도 정상 20회, A~F 이상 179회 및 상부 232g 조건 19회는 평가에만 사용합니다. 원래 V3 전체는 668회입니다.
- 정상 450회는 연속 블록 b01~b05 학습, b06~b07 검증, b08~b09 홀드아웃으로 나뉩니다. 전체 세션이 완전히 분리된 독립 날짜 검증은 아닙니다.
- 테스트 데이터로 정규화·임계값을 다시 계산하지 않습니다. 각 정상군의 FPR과 이상 조건 표시율을 함께 봐야 합니다.

출력은 `MODEL_INDEX.csv`, `CYCLE_SCORES.csv`, `SCORE_SUMMARIES.csv`, 실행 설정·입력 해시와 `models/*.joblib`입니다. 기존 출력 폴더를 덮어쓰지 않습니다. `flag_rate`는 해당 그룹의 이상 표시율이며 정상일 때만 FPR로 해석합니다. 파지 실패·이탈 등 실제 이상 라벨은 사용자 관찰 확인이 필요합니다.

## 원시 데이터부터 특징을 계산할 경우

`build_features.py`는 장치에 연결하지 않고 저장된 파일만 읽습니다. 팀에서 공유받은 원시 세션마다 아래 파일이 필요합니다.

```text
세션폴더/
  manifest.json
  sensors/measurements.csv
  sensors/adxl345_1.csv
  robot/single_queries.csv
```

사이클 목록 CSV에는 `cycle_id`, `session_id`, `cohort`, `source`, `No3_begin_ns`, `No3_end_ns`가 있어야 합니다. `source`는 세션 폴더 경로이며 절대 경로 또는 목록 CSV 기준 상대 경로입니다. 시각은 원래 수집 로그의 호스트 monotonic ns를 사용합니다. 목록의 train/validation은 실제 정상으로 확인된 회차만 포함해야 합니다.

```powershell
.\.venv\Scripts\python.exe build_features.py --manifest "C:\공유데이터\cycle_manifest.csv" --output data/v3_features.csv
.\.venv\Scripts\python.exe train_v3.py --features data/v3_features.csv --output results/v3_from_raw
```

원시 필수 컬럼과 입력 구성 설명은 [특징 설명](docs/FEATURES_KO.md)에 있습니다. 수집 품질·실제 파지 성공을 이 스크립트가 인증하지 않습니다. 원시 UART/HID 감사 및 실제 동작 확인을 먼저 마친 회차 목록을 사용합니다.

## 원래 모델 정의

세 모델 모두 **정상 train만 학습**합니다. 이상 유형을 분류하는 지도 학습이 아닙니다.

- Statistical: 표준화된 특징의 제곱 평균.
- Isolation Forest: 200개 트리, seed 20261031, `n_jobs=1`. 점수는 `-score_samples`.
- PCA: 정상 train에서 분산 95%를 보존하고 표준화 공간의 평균 제곱 재구성 오차를 사용합니다. 모든 차원을 보존한 PCA는 `invalid_PCA`로 표시합니다.
- 공통 판정은 `score > threshold`입니다. 점수 크기는 알고리즘 간 직접 비교하지 않습니다.

11개 특징 구성 × 3개 알고리즘으로 33개 모델을 만듭니다. `V3-A`의 A는 입력 구성 이름이며 A 하중 이상 라벨과는 별개입니다. 이 버전의 관절 위치는 약 2Hz 응답 raw 값이며 목표값 대비 추종 오차나 실제 속도가 아닙니다.

사이클 종료 후 특징을 사용하는 이상 탐지입니다. 실시간 조기 탐지·정확한 고장 원인 진단·미래 고장/RUL 예측 성능을 의미하지 않습니다. 베이스 고정·시작 자세·날짜·반복 이력에 따라 정상 점수도 달라질 수 있습니다. 평가 자료는 개발 중 열람한 이력이 있으며 최종 독립 검증 성능으로 해석하지 않습니다.
