# V5.2와 정상 전용 모델 비교

2026년 11월 11일 No.3 볼트 장착 연구 결과입니다. 기준 모델은 V4 PCA-Q이며, 저장된 V5.2는 준비구간을 제외하고 다시 적합한 비교 후보입니다.

| 실험 | 보존 모델 | 결과 |
|---|---:|---|
| V5.2 준비구간 제외 | PCA 3개 | [보고서](V52_REPORT_KO.md) · [모델 색인](MODEL_INDEX.csv) · [회차별 특징](RAW_FEATURE_REPLAY.csv) |
| V4 특징에 Isolation Forest 적용 | IF 2개 | [PCA-Q·IF·PCA+IF 비교](../v4-pca-if/REPORT_KO.md) |
| 정상 전용 LOF·One-Class SVM·kNN | 주 실험 7개, 두 그룹 분할 8개 | [알고리즘 비교](../normal-only-benchmark/ALGORITHM_COMPARISON_REPORT_KO.md) |

## 저장 모델 점수 재현

Git LFS 파일을 받은 뒤 저장소 루트에서 실행합니다. 원래 Python 3.13과 [고정 패키지 버전](../../requirements.txt)을 사용합니다. 각 모델은 학습 때 정한 특징 순서와 저장 임계값을 사용합니다.

```powershell
git lfs pull
python -m pip install -r requirements.txt
python models/score_followup.py models/v5.2/models/V52_B0_E2_Q.joblib models/v5.2/RAW_FEATURE_REPLAY.csv v52_scores.csv
```

출력은 회차 ID, 점수, 임계값, 초과 여부입니다. V5.2의 준비구간 보류와 평가 분할은 [준비구간 목록](PREPARATION_MASK.csv) 및 [기존 평가표](metrics/CYCLE_SCORES.csv)에서 확인합니다. 위 명령은 입력에 있는 모든 회차에 점수를 계산합니다.

보존한 원래 실험 코드는 이전 버전의 특징·감사 파일과 전체 원본 프로젝트 구조를 전제로 합니다. 위 점수 재현기는 저장소의 모델과 특징 CSV로 실행됩니다. 공개용 표의 로컬 절대경로를 원래 프로젝트 기준 상대경로로 바꿨습니다. 원래 실험의 해시는 당시 입력을 지칭합니다. 공개본 [SHA-256 목록](RELEASE_ARTIFACT_SHA256.csv)과 [모델 점수 검증](RELEASE_MODEL_VERIFICATION.json)을 따로 보존합니다.
