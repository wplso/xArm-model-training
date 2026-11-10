# xArm 41g V1–V5.1 저장 모델

Hiwonder xArm의 No.3 41g 운반 실험에서 저장한 정상 기반 모델 아카이브입니다. 버전별 특징, 학습·평가 분할, 설정, 임계값, 선택한 평가 결과를 함께 정리했습니다.

| 버전 | 파일 수 | 내용 |
|---|---:|---|
| V1 | 6 | 초기 41g 외부 센서·다중 센서 Statistical, Isolation Forest, PCA |
| V1-베이스고정 | 21 | 고정 베이스 정상 데이터 7개 입력 구성 × 3 알고리즘 |
| V2 | 42 | 다중 세션 적합 전략 A/B × 7개 입력 구성 × 3 알고리즘 |
| V2.1 | 42 | 추가 정상 확인 반영, 같은 두 전략과 구성 |
| V3 | 33 | 구간 특징 포함 11개 입력 구성 × 3 알고리즘 |
| V3.1 | 54 | 하중 관련 특징을 포함한 18개 입력 구성 × 3 알고리즘 |
| V4·V4.1·V4.2-A/B | 4 | ADXL 볼트 고정 정상 자료로 적합한 PCA |
| V5 | 15 | ID2 특징 변경을 포함한 PCA 표현 15개와 Q·T²·JOINT 정책 45개 |
| V5.1 | 19 | 다중 세션 적합·교정 비교에서 새로 적합한 PCA 표현, 기존 V4/V5 번들 참조 별도 |

- [MANIFEST.csv](MANIFEST.csv)에서 모델마다 원본 및 공유 SHA-256, 특징 순서, 임계값, 저장 파일을 확인할 수 있습니다.
- `MODEL_INDEX.csv`는 버전별 실행 정책과 모델 경로를 연결합니다. V5의 세 정책이 한 파일을 공유하며, V5.1의 P0/P1은 이전 번들을 참조하고 P2/P3은 하나의 새 PCA를 공유할 수 있습니다.
- `CYCLE_FEATURES.csv`는 저장된 모델의 입력을 확인하고 점수를 재현할 수 있는 회차별 특징입니다. 모델에 넣는 컬럼과 순서는 각 저장 모델의 `features`를 사용합니다. 날짜·세션·조건과 `dataset_session_path`는 자료 연결용입니다. 경로가 비어 있으면 `session_id`로 원본 공유 폴더의 세션 목록을 확인합니다.
- 특징 표와 분할 CSV의 `project-source/` 항목은 원본 프로젝트에서의 출처 이름입니다. 공유 원시 데이터의 실제 경로는 `dataset_session_path`와 원본 공유 폴더의 `SESSION_INDEX.csv` 또는 `세션_목록.csv`를 사용합니다. 출처 메타데이터를 정리한 CSV/JSON과 원본 사이의 해시는 [ARTIFACT_SHA256.csv](ARTIFACT_SHA256.csv)에서 구분합니다.
- V1의 초기 500회 중 실제 41g 조건이 불확실한 마지막 50회는 특징 표에서 제외하고 라벨 감사 표에 남겼습니다. V4 특징 표는 물리 조건과 계측 품질이 확인된 399회로 구성됩니다.

## 불러오기와 검증

원래 검증 환경은 Python 3.13.15, NumPy 2.5.4, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.9.1, joblib 1.6.0입니다. 저장 모델은 저장소에 동봉한 `model_runtime/normal_models.py`의 `normal_models.Detector` 클래스를 참조합니다. 모델을 읽기 전에 해당 디렉터리를 Python 모듈 탐색 경로에 추가하세요. 신뢰하는 출처의 joblib 파일만 역직렬화하세요.

```powershell
git lfs pull
python -m pip install -r requirements.txt
python -B -X utf8 models/verify_models.py --checksums-only
python -B -X utf8 models/verify_models.py
```

원래 Windows 분석 환경에서 점수의 비트 단위 재현도 확인할 수 있습니다.

```powershell
python -B -X utf8 models/verify_models.py --exact-scores
```

`detector.score`의 값이 커질수록 정상 학습 분포에서 멀어집니다. 일반 번들은 `threshold`에 대해 엄격한 `>`를 사용합니다. V5 T²·JOINT는 [점수 정의](v5/SCORE_DEFINITIONS.json)의 계산을 적용한 뒤 정책별 `thresholds`를 사용합니다. V5.1은 `MODEL_INDEX.csv`의 분할·정책별 임계값을 사용합니다. 검증 도구의 저장 점수 해시는 공통 `detector.score` 결과를 확인합니다.

V5.1 번들은 원래 장비 PC의 경로를 담은 `input_hashes` 사전의 키만 출처 상대 경로로 바꿨습니다. 모델 값, 특징 순서, 교정 회차와 임계값은 보존했습니다. 다른 버전의 모델은 파일 그대로 보존했습니다. 점수 재현은 기존 데이터와 모델의 일치 여부를 확인합니다. 실시간 탐지 또는 독립 현장 평가를 의미하지 않습니다.
