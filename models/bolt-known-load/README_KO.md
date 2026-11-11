# 볼트 장착 하중 지도학습과 독립 평가

2026년 11월 11~12일 No.3 운반에서 정상은 41g 운반, A121은 121g 물체 운반, B82·C182는 41g 운반 중 각각 상부 82g·하부 182g 추가 조건입니다. 아래 결과는 저장된 회차 종료 특징을 사용합니다.

| 실험 | 결과와 산출물 |
|---|---|
| 기존 몬스터겔 학습 → 볼트 세션 홀드아웃 | [4클래스 로지스틱 회귀·Random Forest](supervised/REPORT_KO.md) · [기존 장착의 재현용 특징](supervised/HISTORICAL_FEATURES.csv) |
| 이진 하중 경보 | [이진 로지스틱 회귀·4클래스 비교](binary/REPORT_KO.md) |
| 볼트 학습 및 동결 | [동결 절차와 세 모델](prospective/HANDOFF_KO.md) · [동결 명세](prospective/FROZEN_MODELS.json) |
| 교차 순서 A/B/C 평가 180회 | [동결 세 모델 평가](abc-interleaved/frozen_evaluation/REPORT_KO.md) · [A121 기여도](abc-interleaved/bolt_a121_attribution/REPORT_KO.md) |
| 정상 240회 추가 수집 및 정상 확장 로지스틱 | [새 모델과 학습·교정·홀드아웃](retrain/retrained_bolt_normal/REPORT_KO.md) · [가상 임계값 비교](retrain/threshold_and_a121/THRESHOLD_REPORT_KO.md) · [A121 특징 제거 실험](retrain/a121_single_feature_ablation_20261112/REPORT_KO.md) |
| B82·C182 독립 평가 90회 | [고정 임계값 비교](bc-independent/REPORT_KO.md) · [사전 계획](bc-independent/CAPTURE_PLAN.json) |

교차 순서 평가에서 동결 V4 PCA-Q, 몬스터겔 로지스틱, 볼트 로지스틱의 정상 오탐은 각각 19·2·14/90회, 하중 탐지는 44·50·59/90회였습니다. 정상 확장 후보는 N01·N02 120회를 학습에 추가했고 N03 60회로 교정했습니다. N04 60회는 정상 평가에 사용했습니다.

| B82·C182 독립 평가 | 저장 임계값 0.821445 | 비교 임계값 0.40 |
|---|---:|---:|
| 정상 오탐 | 3/30 | 6/30 |
| B82 탐지 | 23/30 | 26/30 |
| C182 탐지 | 23/30 | 28/30 |

공통 정상 30회 대비 AUROC는 B82 0.931, C182 0.958입니다. 두 임계값을 같은 모델의 같은 점수에 적용했습니다. 별도 B82·C182 90회는 평가 전용이며 A121의 독립 재검증은 포함하지 않습니다.

## 원시 데이터와 점수 재현

[원시 데이터 안내](../../원본%20수집%20데이터/볼트_하중_보완원시/README.md)에 세 캠페인의 24개 세션과 510회의 원시 CSV·회차·역할 목록이 있습니다.

Git LFS 파일과 [고정 패키지](../../requirements.txt)를 설치한 뒤 저장소 루트에서 실행합니다. `--threshold 0.40`은 비교용 출력에만 적용됩니다. 이 옵션을 생략하면 저장 모델의 임계값을 사용합니다.

```powershell
git lfs pull
python -m pip install -r requirements.txt
python models/score_followup.py models/bolt-known-load/retrain/retrained_bolt_normal/models/LOGISTIC_BOLT_NORMAL_EXPANDED.joblib models/bolt-known-load/bc-independent/CYCLE_FEATURES.csv bc_scores.csv --threshold 0.40
```

원래 실험 코드는 전체 수집 프로젝트의 입력 구조와 당시 감사 해시를 사용합니다. 공개 모델의 [SHA-256 목록](RELEASE_ARTIFACT_SHA256.csv)과 [저장 점수 및 경보 검증](RELEASE_MODEL_VERIFICATION.json)에는 배포 파일에 대한 검증 결과가 있습니다. 동결 비교 폴더에는 V4 PCA-Q와 몬스터겔 로지스틱의 동일 바이트 사본도 있습니다.
