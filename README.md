# xArm 정상 기반 이상 탐지

Hiwonder xArm의 No.3 작업에서 41g 블록을 운반하고 자동으로 원복하는 동안 전류·전압, 가속도, 실제 서보 위치를 수집했습니다. 모델은 주로 No.3 운반 구간을 요약해 이상을 판정합니다. `main`에는 **V3 학습 코드**가 있으며, 아래는 후속 버전을 포함한 평가 결과입니다.

**현재 정상 기반 비교 기준은 V4 PCA-Q입니다.** V5~V5.2는 연구 비교로 보존했습니다. 볼트 고정 로지스틱은 알려진 하중을 이용한 지도학습 후보이며, 0.40은 B82·C182 독립 평가에 사용한 후보 임계값입니다.

[V3 실행 방법](#환경과-실행) · [V3.1 학습과 데이터](https://github.com/wplso/xArm-model-training/tree/v3.1) · [V1~V5.1 저장 모델](https://github.com/wplso/xArm-model-training/blob/v5.1/models/README_KO.md) · [V5.2와 알고리즘 비교](https://github.com/wplso/xArm-model-training/tree/v5.2) · [볼트 하중 모델과 신규 데이터](https://github.com/wplso/xArm-model-training/tree/bolt-known-load)

## 버전별 대표 성능

수치는 각 모델의 고정 임계값에서 **오탐 또는 탐지한 회차 / 평가 회차**입니다. 평가 세션·하중·센서 장착이 다른 행을 순위처럼 비교하면 안 됩니다.

### 초기 41g 및 테이프 장착 평가

| 대표 모델 | 평가 조건 | 정상 오탐 | 이상 탐지 |
|---|---|---:|---|
| V1 통합 PCA / Isolation Forest | 41g 같은 날짜 후반 정상 | PCA 0/100, IF 9/100 | 정상만 평가 |
| V2 A7 IF B | 공통 홀드아웃, 82g 운반 / 상부 82g / 하부 82g | 27/200 | 5/20 · 15/20 · 2/20 |
| V2.1 A7 IF B | V2와 동일한 평가 자료 | 26/200 | 6/20 · 19/20 · 1/20 |
| V1-베이스고정 A7 IF | 연속 수집 후반 및 별도 정상, 82g 운반 / 상부 82g / 하부 82g | 5/100 · 9/20 | 5/30 · 14/30 · 1/30 |
| V3-A PCA | 11월 5일 A~F 평가 | 16/70 | A/B/C 16/20 · 18/20 · 12/20, D/E/F 각 20/20. Macro-AUROC 0.915 |
| V31-15 PCA | V3와 동일한 A~F 평가 | 24/70 | A/B/C 16/20 · 15/20 · 5/20, D/E/F 각 20/20. Macro-AUROC 0.818 |

11월 5일과 아래 볼트 장착 평가에서 A는 121g 물체 운반, B는 41g 운반과 상부 82g 부착, C는 41g 운반과 하부 182g 부착입니다. D/E/F는 최초 파지 실패, 들어 올릴 때 이탈, 도달 범위 밖 위치 오류입니다. [세부 평가 및 초기 무부하 실험](https://github.com/wplso/xArm-model-training/blob/v5.1/README.md#모델-버전과-대표-평가-결과)

### ADXL345 볼트 장착 평가

| 대표 모델 | 정상 오탐 | A 탐지 | B 탐지 | C 탐지 |
|---|---:|---:|---:|---:|
| [V4 PCA](https://github.com/wplso/xArm-model-training/tree/v4) | 10/99 | 10/30 | 12/30 | 21/30 |
| [V4.1 PCA](https://github.com/wplso/xArm-model-training/tree/v4.1) | 5/99 | 5/30 | 4/30 | 14/30 |
| V4.2-A PCA | 5/99 | 4/30 | 6/30 | 16/30 |
| V4.2-B PCA | 11/99 | 10/30 | 5/30 | 21/30 |
| [V5 B0 F3 Q](https://github.com/wplso/xArm-model-training/tree/v5) | 11/219 · 재가동 20/40 | 6/30 | 10/30 | 23/30 |
| V5 B2 F1 Q | 12/219 · 재가동 16/40 | 4/30 | 12/30 | 19/30 |
| [V5.1 B1 P2](https://github.com/wplso/xArm-model-training/tree/v5.1) | 10/160 · 재가동 5/40 | 5/30 | 0/30 | 8/30 |

- V4 계열의 정상 99회는 같은 연속 운전의 홀드아웃입니다. V5의 219회는 이 99회와 이후 정상 120회를 합친 개발 평가이며, 재가동 40회는 별도입니다. V4를 같은 두 자료에 적용하면 각각 22/219와 23/40입니다. [V5 평가 보고서](https://github.com/wplso/xArm-model-training/blob/v5/models/v5/V5_REPORT_KO.md)
- V5.1의 B1 P2는 같은 특징으로 분할별 정상 학습을 확장한 구성입니다. 정상 160회는 세 분할의 평가 회차를 한 번씩 집계했습니다. 기존 V4.1을 고정한 B1 P0의 오탐은 17/160, 재가동 10/40입니다. A/B/C는 관련 정상 세션까지 학습·교정에서 제외한 별도 회고 평가이며, 이 조건에서 P0의 탐지는 각각 5/30, 4/30, 14/30입니다. [V5.1 평가 보고서](https://github.com/wplso/xArm-model-training/blob/v5.1/models/v5.1/V51_REPORT_KO.md)

## V5.2 이후 실험

### 정상만으로 학습한 모델

- [V5.2 준비구간 제외](https://github.com/wplso/xArm-model-training/blob/v5.2/models/v5.2/V52_REPORT_KO.md). 준비구간으로 정상 40/259회와 하중 30/90회의 판정을 보류했습니다. 남은 동일 회차에서 V4 24특징 PCA-Q의 정상 오탐은 기존 33/219회에서 재학습 후 35/219회였습니다. A/B/C 탐지는 각각 7→4, 7→5, 14→16회(각 20회)였습니다.
- [PCA+IF](https://github.com/wplso/xArm-model-training/blob/v5.2/models/v4-pca-if/REPORT_KO.md)와 [LOF·One-Class SVM·kNN](https://github.com/wplso/xArm-model-training/blob/v5.2/models/normal-only-benchmark/ALGORITHM_COMPARISON_REPORT_KO.md). 동일한 V4 24개 특징을 사용했습니다. IF 단독은 재가동 정상 오탐을 V4 PCA-Q의 23/40회에서 11/40회로 낮췄으나 A/B/C 탐지는 10·12·21회에서 1·3·2회(각 30회)로 줄었습니다. 재가동 오탐 감소와 세 하중 탐지 유지가 함께 확인된 후보는 없었습니다.

### 볼트 고정 하중 지도학습

11월 11일 새 볼트 고정 캠페인의 같은 180회에서 동결 모델을 비교했습니다. 정상 90회와 A121·B82·C182 각 30회입니다. 수치는 하중 유형 일치 여부와 관계없이 경보가 난 횟수입니다. [동결 모델 평가](https://github.com/wplso/xArm-model-training/blob/bolt-known-load/models/bolt-known-load/abc-interleaved/frozen_evaluation/REPORT_KO.md)

| 모델 | 정상 오탐 | A121 탐지 | B82 탐지 | C182 탐지 |
|---|---:|---:|---:|---:|
| V4 PCA-Q | 19/90 | 14/30 | 9/30 | 21/30 |
| 몬스터겔 자료 학습 4클래스 로지스틱 | 2/90 | 4/30 | 23/30 | 23/30 |
| 볼트 고정 자료 학습 4클래스 로지스틱 | 14/90 | 12/30 | 24/30 | 23/30 |

정상 학습 범위를 확장한 볼트 로지스틱은 별도 정상 60회에서 오탐이 기존 볼트 모델 8회에서 1회로 줄었습니다. 이미 분석한 하중 자료에서는 A/B/C 탐지가 12·24·23회에서 2·13·8회(각 30회)로 줄었습니다. [정상 확장 실험](https://github.com/wplso/xArm-model-training/blob/bolt-known-load/models/bolt-known-load/retrain/retrained_bolt_normal/REPORT_KO.md)

11월 12일에는 새 정상·B82·C182 각 30회에 **같은 정상 확장 모델의 점수**를 사용했습니다. 두 임계값은 수집 전에 고정했습니다. [독립 임계값 검증](https://github.com/wplso/xArm-model-training/blob/bolt-known-load/models/bolt-known-load/bc-independent/REPORT_KO.md)

| 지표 | 저장 임계값 0.821445 | 후보 임계값 0.40 |
|---|---:|---:|
| 정상 오탐 | 3/30 (10.0%) | 6/30 (20.0%) |
| B82 탐지 | 23/30 (76.7%) | 26/30 (86.7%) |
| C182 탐지 | 23/30 (76.7%) | 28/30 (93.3%) |

공통 정상 대비 AUROC는 B82 0.931, C182 0.958입니다. 0.40은 이번 캠페인에서 하중을 8회 더 탐지하고 정상 오탐을 3회 늘렸습니다. A121은 이번 90회에 포함되지 않았고, 새 회차는 평가에만 사용했습니다.

### 추가 원시 자료

[볼트 하중 보완 원시 데이터](https://github.com/wplso/xArm-model-training/tree/bolt-known-load/%EC%9B%90%EB%B3%B8%20%EC%88%98%EC%A7%91%20%EB%8D%B0%EC%9D%B4%ED%84%B0/%EB%B3%BC%ED%8A%B8_%ED%95%98%EC%A4%91_%EB%B3%B4%EC%99%84%EC%9B%90%EC%8B%9C)는 24개 수집 세션의 510회입니다. 교차 A/B/C 평가 180회, 정상 확장 240회, B/C 독립 평가 90회입니다. 세션별 전류·전압, 가속도, 서보 위치 CSV와 No.3·원복 시간 경계, 조건 라벨, 학습·교정·평가 역할, 파일 해시를 함께 제공합니다. CSV는 Git LFS로 내려받습니다.

## 현재 한계

- **정상 세션에 따른 오탐 차이**. 같은 날 41g을 운반한 두 세션에서 V4의 오탐은 정상 40회 중 5회, 전원 재가동 후 정상 40회 중 23회였습니다. 센서 장착과 시작 자세·재가동 이력이 정상 점수에 영향을 줄 수 있습니다. 휴지 실험만으로 워밍업·온도와 운전 이력의 영향을 분리하지 못했습니다.
- **가벼운 하중 탐지**. 정상 범위를 넓히면 일부 하중 이상 탐지도 약해졌습니다. PCA 재구성 잔차(Q)와 주성분 공간 안의 변화(T²)를 비교했으며, 두 점수를 결합할 운영 기준은 아직 확정하지 않았습니다.
- **알려진 하중의 적용 범위**. B82·C182의 0.40 결과는 새 볼트 캠페인 한 건에서 확인했습니다. 날짜를 더 바꾼 재현성, A121 점수 분리, 자연 고장과 미학습 이상에 대한 성능은 계속 확인이 필요합니다.
- **계측 및 검증 범위**. 로봇 전체 전류·전압, ADXL345 한 개, 서보 위치를 기록합니다. 관절별 전류·온도·입력 전압은 미확보입니다. 추가 날짜의 동일 장착 재현성과 볼트 장착 D/E/F 평가가 남아 있습니다. 현재 결과는 사이클 종료 후 이상 탐지이며, 실시간 조기 경보·고장 원인 진단·수명 예측은 검증 범위 밖입니다.

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

V3 재현용 특징 CSV와 원시 자료는 `v3.1` 브랜치의 `원본 수집 데이터/`에서 Git LFS로 받을 수 있습니다. V1~V5.1 저장 모델은 `v5.1`에, V5.2 이후 실험과 새 볼트 데이터는 위 버전별 브랜치에 있습니다. [데이터 받기와 학습 방법](https://github.com/wplso/xArm-model-training/blob/v3.1/README.md#데이터-받기와-바로-학습)

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
