# xArm 이상 탐지

Hiwonder xArm AI의 No.3 작업에서 로봇 전체 전류·전압, ADXL345 가속도, 서보 6개의 실제 위치를 기록합니다. 정상 조건은 41g 블록 운반과 자동 원복입니다.

## Raspberry Pi 4B 실시간 성능 · 11월 14~15일

Pi에서 41g 블록 No.3 운반과 자동 원복을 세 차례 시험했습니다. No.3 시작 2초 뒤부터 수집·정렬·GRU 추론·저장을 병행했고, 로지스틱 S2와 이후 추가한 CatBoost S2는 No.3 종료 후 점수를 계산했습니다.

| 실기 10회씩 | 실제 운반·원복 성공 | GRU 발행 / 예정 | 예정 시각부터 기록까지 P95 | 실제 발행 간격 P95 / 최대 |
| --- | ---: | ---: | ---: | ---: |
| 초기 설정 | 9/10 | 513/513 | 48.56ms | 202.05 / 239.28ms |
| CatBoost S2 점수 기록 추가 | 9/10 | 515/516 | 52.76ms | 202.31 / 243.79ms |
| 관절 응답 직전 5ms 허용 | 10/10 | 516/516 | 53.96ms | 205.29 / 238.07ms |

기록된 발행 중 예정 시각 기준 200ms 마감 초과는 0건이었습니다. 유일한 발행 누락은 첫 입력 창에 전류·전압 표본이 아직 없던 경우였습니다. No.3 동작 중 위치 조회 타임아웃은 0건이었고, CatBoost 병행 시험에서 원복 뒤 ID3 확인이 한 번 타임아웃됐습니다. 최신 실기의 위치 조회 타임아웃은 원복까지 포함해 0건이었습니다. 전원 교체 후 저전압·스로틀링 상태는 `0x0`이었습니다.

최신 정상 10회에서 실시간 GRU 단발 경보는 3회(전류 1회, 집게 위치 2회)였고 로지스틱 S2와 CatBoost S2는 각각 0회였습니다. 이전 두 실기에서는 실제 파지 실패를 한 번씩 관찰했고, GRU가 두 회 모두 경보했습니다. CatBoost는 점수 기록용으로 실행했고 최신 실기에서 회차당 추론 P95는 약 11.5ms였습니다.

직전 5ms 허용으로 이전 CatBoost 병행 로그의 GRU 오탐은 3/9에서 1/9로 줄었고, 실제 파지 실패 경보는 유지됐습니다. 최신 실기 로그에서는 기존 정렬과 5ms 정렬이 동일한 3/10 오탐을 냈습니다.

## GRU 경보 규칙 비교 · 0.5초와 0.65초 확인 창

11월 13일 평가 로그 240회와 Pi 실기 로그 30회에서 동결된 GRU F2·O3 ALL, 임계값 7.288925, 관절 응답 직전 5ms 허용 규칙을 사용했습니다. 확인 창은 같은 회차에서 서로 다른 시각에 발생한 초과 2건의 간격으로 계산했습니다. 센서나 관절이 달라도 같은 창에 포함했습니다.

| 11월 13일 평가 유형 | 1회 초과 | 0.5초 내 2회 | 0.65초 내 2회 | 1회 초과 대비 추가 지연 중앙값 · 0.5초 → 0.65초 |
| --- | ---: | ---: | ---: | ---: |
| 정상 오탐 | 1/90 | 0/90 | 0/90 | — |
| D · 최초 파지 실패 | 30/30 | 29/30 | **30/30** | 1.066 → 0.616초 |
| E · 물체 이탈 | 30/30 | 30/30 | 30/30 | 1.573 → 0.500초 |
| F · 물체 위치 오류 | 30/30 | 30/30 | 30/30 | 1.066 → 0.616초 |
| 폼 충돌 | 30/30 | 30/30 | 30/30 | 0.200 → 0.200초 |
| 작업대 흔들림 | 30/30 | 30/30 | 30/30 | 0.042 → 0.042초 |

0.65초 창에서 회복된 D는 ID1과 ID6 초과가 618ms 떨어진 회차로, No.3 시작 5.126초에 확인됐습니다. 지연은 초과 1회 기준과 각 확인 창에서 모두 탐지한 회차별 시각 차이입니다. 최대 추가 지연은 0.5초 창에서 2.068초(F), 0.65초 창에서 1.594초(E)였습니다.

Pi 로그를 같은 5ms 규칙으로 재생하면 정상 28회의 오탐은 두 확인 창에서 모두 0회였습니다. 실제 파지 실패는 두 창 모두 2/2회 탐지됐고, 0.65초 창에서 각각 약 0.450초 빨리 확인됐습니다(5.573초 → 5.123초). 지연은 저장된 관측값의 가용 시각 기준입니다. 두 번 초과하는 규칙은 오프라인으로 비교했으며, Pi의 실시간 경보는 기존 1회 초과 규칙을 사용합니다.

## 대표 성능

**PC 후보는 TabPFN-v2 S2 P0 OR GRU F2·O3 ALL**, **Raspberry Pi 4B 후보는 로지스틱 S2 OR 같은 GRU**입니다. TabPFN과 로지스틱은 회차 종료 후 알려진 하중 이상을 판정하고, GRU는 동작 중 다중 센서 예측과 실측의 차이로 경보합니다. 같은 회차에서 둘 중 하나라도 경보하면 탐지로 집계했습니다.

| 공통 평가 370회 | PC 후보 | Raspberry Pi 4B 후보 |
|---|---:|---:|
| 정상 오탐 | **3/160 (1.88%)** | **6/160 (3.75%)** |
| 이상 탐지 | **209/210 (99.52%)** | **208/210 (99.05%)** |
| 상부 프레임 하중 B82 | 29/30 | 29/30 |
| 하부 프레임 하중 C182 | 30/30 | 29/30 |
| 최초 파지 실패 D · 물체 이탈 E · 도달 범위 밖 F · 폼 접촉 · 작업대 흔들림 | 각각 30/30 | 각각 30/30 |

이미 확보한 공통 평가 로그를 동결 모델과 기존 임계값으로 재생한 결과입니다. A121은 이 평가에 포함되지 않았습니다. 새 독립 이상 캠페인과 Pi에서의 두 번 초과 규칙은 실기 검증이 남아 있습니다.

[모델별 성능](모델별%20성능.md) · [기준 모델과 고정 설정](https://github.com/wplso/xArm-model-training/tree/reference-models/models/reference-models) · [370회 재생 상세](https://github.com/wplso/xArm-model-training/blob/reference-models/models/reference-models/joint-replay/REPORT_KO.md) · [공유 원시 데이터](https://github.com/wplso/xArm-model-training/tree/reference-models/%EC%9B%90%EB%B3%B8%20%EC%88%98%EC%A7%91%20%EB%8D%B0%EC%9D%B4%ED%84%B0)

Built with PriorLabs-TabPFN.

## V3 학습 코드

`main`에는 V3 학습 코드가 있습니다. [V3.1 학습 코드와 원시 데이터](https://github.com/wplso/xArm-model-training/tree/v3.1)도 볼 수 있습니다.

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

V3 재현용 특징 CSV와 원시 자료는 `v3.1` 브랜치의 `원본 수집 데이터/`에서 Git LFS로 받을 수 있습니다. V1~V5.1 저장 모델은 `v5.1`에, V5.2 이후 실험과 새 볼트 데이터는 각 버전 브랜치에 있습니다. 위의 모델별 성능 문서에서 확인할 수 있습니다. [데이터 받기와 학습 방법](https://github.com/wplso/xArm-model-training/blob/v3.1/README.md#데이터-받기와-바로-학습)

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

사이클 종료 후 특징을 사용하는 이상 탐지입니다. 실시간 조기 탐지·정확한 고장 원인 진단·미래 고장/RUL 예측 성능을 의미하지 않습니다. 베이스 고정·시작 자세·날짜·반복 이력에 따라 정상 점수도 달라질 수 있습니다. 이 절의 V3 평가 자료는 개발 중 열람한 이력이 있으며 최종 독립 검증 성능으로 해석하지 않습니다.
