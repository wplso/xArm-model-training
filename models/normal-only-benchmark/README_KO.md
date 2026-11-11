# xArm 정상 전용 알고리즘 고정 비교

볼트·너트로 고정한 ADXL345, 고정 베이스, No.3 41g 운반을 기준으로 한 오프라인 개발 비교다. V4의 입력 24개를 같은 순서로 사용한다. 주 실험은 정상 200회 학습, 정상 100회 교정, 정상 259회와 하중 이상 90회 평가로 구성한다. 평가 정상은 전량 사용하고, 재가동 11~40회와 21~40회는 같은 점수의 부분집합으로 집계한다.

## 실행 환경

프로젝트의 기존 Windows 가상환경을 사용했다. Python 3.13.15, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, SciPy 1.18.1, joblib 1.6.0이다. 저장 모델은 동일 계열의 설치 버전에서 열고, 로컬에서 검증한 joblib 파일을 사용한다. 모델 색인의 시간은 이 Windows PC에서 이미 추출된 24개 특징 1행을 대치·표준화·채점하는 시간이다. 실시간 수집·특징 계산·라즈베리파이에서의 실행은 시간에 포함되지 않는다.

PowerShell에서 프로젝트 루트와 이 폴더를 지정한다. 재실행할 때는 쓰기 가능한 새 결과 폴더를 `$out`으로 지정한다.

```powershell
$project = (Get-Location).Path
$study = Join-Path $project 'hardware\xarm_model_versions\normal_only_algorithm_benchmark'
$python = Join-Path $project '.venv\Scripts\python.exe'
$out = Join-Path (Get-Location) 'normal_only_algorithm_benchmark_replay'

& $python -B -X utf8 "$study\analysis_code\run_benchmark.py" --project-root $project --output $out
& $python -B -X utf8 "$study\analysis_code\summarize_benchmark.py" --project $project --study $out
& $python -B -X utf8 "$study\analysis_code\neighbor_diagnostics.py" --project-root $project --run-dir $out
& $python -B -X utf8 "$study\analysis_code\plot_benchmark.py" --study-dir $out
```

요청서에 모델과 수치를 먼저 지정했다. 집계 코드의 교차 확인에서 설정 JSON 저장 전에 동일한 지정값으로 메모리 계산을 한 차례 수행했다. 결과를 보존하는 실행에서는 `EXPERIMENT_CONFIG.json`을 먼저 썼고, 두 계산 사이에 모델 설정이나 평가 규칙을 변경하지 않았다.

## 입력과 판정

- `FEATURE_SCHEMA.json`에 특징 24개의 순서가 있다. `DATA_USAGE_MANIFEST.csv`는 고유 649회, `SPLIT_MANIFEST.csv`는 주 비교와 두 보조 분할의 회차 역할을 기록한다.
- 주 실험은 저장된 V4 대치기와 표준화기를 재사용한다. 보조 분할의 대치기·표준화기는 각 분할의 학습 정상만 사용해 적합한다. 새 모델은 모두 같은 24차원 표준화 행렬을 받는다.
- 새 모델은 정상 교정 점수의 95백분위수(`method="higher"`)를 경계로 사용한다. `margin = score - threshold`가 양수일 때 경보다. 원점과 척도가 모델마다 달라 margin의 크기는 동일 모델 안에서 비교한다.
- LOF는 novelty 점수로 교정·평가 회차를 읽고, kNN은 학습 집합의 최근접 20개 유클리드 거리 평균을 계산한다. SVM 점수는 `-score_samples`이며, gamma의 기준은 학습 표준화 행렬의 분산에서 계산한다. [LOF 공식 문서](https://scikit-learn.org/1.9/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) · [One-Class SVM 공식 문서](https://scikit-learn.org/1.9/modules/generated/sklearn.svm.OneClassSVM.html) · [NearestNeighbors 공식 문서](https://scikit-learn.org/1.9/modules/generated/sklearn.neighbors.NearestNeighbors.html)

## 결과 읽는 순서

1. [비교 보고서](ALGORITHM_COMPARISON_REPORT_KO.md)에서 정상 세션별 오탐과 하중별 탐지를 확인한다.
2. [기준선 재현](metrics/BASELINE_REPRODUCTION.csv)과 [주 비교](metrics/PRIMARY_MODEL_COMPARISON.csv)를 대조한다.
3. [정상 세션](metrics/NORMAL_SESSION_METRICS.csv), [짝 정상과 이상](metrics/PAIRED_ANOMALY_METRICS.csv), [경보 전이](metrics/PAIRED_ALARM_TRANSITIONS.csv)를 보고 어떤 회차의 판단이 바뀌었는지 확인한다.
4. [설정 민감도](metrics/PARAMETER_SENSITIVITY.csv), [이웃 진단](metrics/NEIGHBOR_DIAGNOSTICS.csv), [수치 진단](metrics/NUMERICAL_DIAGNOSTICS.csv), [그룹 제외 확인](metrics/GROUP_HELDOUT_COMPARISON.csv)은 판단 방식의 차이를 설명한다. 이웃 겹침은 짝 정상과 이상이 선택한 학습 참조 ID의 상대 빈도에서 공통 부분을 합한 값이다.
5. [모델 색인](MODEL_INDEX.csv), [회차별 점수](metrics/CYCLE_SCORES.csv), [검증](metrics/VERIFICATION.json)에 모델 객체와 판정 재현 근거가 있다.

AUROC는 짝 정상과 이상 점수의 순위 분리를, TP/FP는 교정으로 고정한 경계에서의 결과를 나타낸다. AP는 각 조건의 양성 비중에 영향을 받는다. A121의 짝 정상은 30회, B82와 C182는 각각 10회다. Test99는 학습과 같은 연속 운전의 후반이며, 이번에 사용한 정상·이상 자료는 반복 열람한 개발 자료다.
