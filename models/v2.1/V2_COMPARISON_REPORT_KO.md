# xArm 41g 다중 세션 Baseline V2.1

## 데이터 및 분할

주 정상 1,230회 가운데 짝 정상 30회는 별도 평가다. 나머지는 학습 후보 850회, 검증 150회, 공통 홀드아웃 200회다. 기존 수동 41g 10회는 예전 82g 비교 전용이다.

- A 학습은 11/01 300회, 11/03 100회와 11/04 신규 450회다.
- B 학습 200회는 동일 학습 블록에서 11/01 100회와 11/03 100회를 사용했다. 각 날짜 안에서 블록 비중을 유지하며 시간순 등간격으로 선택했다.
- 검증 150회는 11/01 b07 50회와 11/03 추가 자동 원복 100회다. 홀드아웃 200회는 11/01 b08–09 100회, 11/03 콜드 20회와 마지막 완료 80회다.
- 신규 450회는 11/04 실제 수집, 11/03 실험 묶음이다. 백업 9개와 계측 감사가 통과했고, 사용자가 450회 모두 41g 운반·원복 정상으로 확인했다. A에 전량 반영했다. B는 11/03~04를 같은 실험 묶음으로 처리해 그중 100회를 선택했다.
- 11/01 b10 50회는 무부하 혼재로 제외했다. 수동 98회, 중단 세션 12회, 오부착 10회는 실제 운반 또는 조건 확인 부족으로 제외했다. 종료 판정에 실패한 회차와 확인된 무부하 회차도 정상 적합에 사용하지 않았다.

V1 6종과 선택 Ablation 21개 조합의 저장 점수·판정·임계값을 재현했다. V2 42개를 저장하고 다시 읽어 전체 점수를 재현했다. V2 A가 V1 검증 b06을 학습에 사용한 사실과 B 균형 미선택 행은 manifest에 표시했다. 모든 V2 전처리는 해당 전략의 정상 학습 자료만 사용했고 임계값은 공통 검증의 95% 분위수(higher)로 고정했다.

## 정상 평가

홀드아웃 200회는 두 모델 모두의 적합과 임계값 결정에서 제외됐다. 과거에 분석한 날짜와 세션이므로 이번 비교는 탐색적이다. 검증 FPR은 임계값을 정한 각 버전의 검증 집단 결과다. 세션 평균 및 가중 FPR과 최대값, 점수 분위수, 시작 자세·복원·운전 이력별 분모는 CSV에 있다.

| model_id | series | validation_n | validation_FPR | holdout_FP | holdout_n | holdout_FPR | worst_session_FPR |
|---|---|---|---|---|---|---|---|
| A2_Statistical | V1-reference | 100 | 4.0% | 41 | 200 | 20.5% | 95.0% |
| A2_Statistical | V2-A | 150 | 4.7% | 19 | 200 | 9.5% | 60.0% |
| A2_Statistical | V2-B | 150 | 4.7% | 20 | 200 | 10.0% | 60.0% |
| A2_IsolationForest | V1-reference | 100 | 4.0% | 41 | 200 | 20.5% | 75.0% |
| A2_IsolationForest | V2-A | 150 | 4.7% | 20 | 200 | 10.0% | 60.0% |
| A2_IsolationForest | V2-B | 150 | 4.7% | 21 | 200 | 10.5% | 60.0% |
| A2_PCA | V1-reference | 100 | 4.0% | 8 | 200 | 4.0% | 6.7% |
| A2_PCA | V2-A | 150 | 4.7% | 6 | 200 | 3.0% | 4.0% |
| A2_PCA | V2-B | 150 | 4.7% | 6 | 200 | 3.0% | 6.7% |
| A7_Statistical | V1-reference | 100 | 4.0% | 45 | 200 | 22.5% | 75.0% |
| A7_Statistical | V2-A | 150 | 4.7% | 26 | 200 | 13.0% | 70.0% |
| A7_Statistical | V2-B | 150 | 4.7% | 24 | 200 | 12.0% | 70.0% |
| A7_IsolationForest | V1-reference | 100 | 4.0% | 61 | 200 | 30.5% | 70.0% |
| A7_IsolationForest | V2-A | 150 | 4.7% | 31 | 200 | 15.5% | 70.0% |
| A7_IsolationForest | V2-B | 150 | 4.7% | 26 | 200 | 13.0% | 65.0% |
| A7_PCA | V1-reference | 100 | 4.0% | 27 | 200 | 13.5% | 46.7% |
| A7_PCA | V2-A | 150 | 4.7% | 2 | 200 | 1.0% | 5.0% |
| A7_PCA | V2-B | 150 | 4.7% | 9 | 200 | 4.5% | 15.0% |

유효 구성의 전략별 중앙값은 아래와 같다. A/B는 표본 수와 날짜 비율이 함께 달라 각각의 효과를 분리할 수 없다.

| strategy | holdout_median | median_worst_session | median_mean_gap |
|---|---|---|---|
| A | 0.095 | 0.6 | 0.1667 |
| B | 0.1 | 0.6 | 0.1833 |

## 이상 평가

최근 조건별 41g 대조 10회와 하중 20회를 사용했다. AP는 하중 조건을 양성으로 계산했다. 시작 자세를 맞춘 보조 비교와 과거 수동 복원 41g/82g 비교를 별도 저장했다. 과거 무부하 프레임 하중은 작업 조건이 달라 41g Recall과 합산하지 않았다. 무부하 상·하부 비교는 같은 대조를 재사용했다. 각 이상 조건의 독립 세션이 부족해 연속 Cycle을 독립으로 가정한 신뢰구간은 산출하지 않았다.

| model_id | version | strategy | pair | FPR | TPR | AUROC | TPR_minus_FPR |
|---|---|---|---|---|---|---|---|
| A2_IsolationForest | V1 | reference | recent_payload82 | 0.7 | 0.55 | 0.59 | -0.15 |
| A2_PCA | V1 | reference | recent_payload82 | 0.0 | 0.0 | 0.71 | 0.0 |
| A2_Statistical | V1 | reference | recent_payload82 | 0.7 | 0.65 | 0.6 | -0.05 |
| A7_IsolationForest | V1 | reference | recent_payload82 | 0.4 | 0.7 | 0.7 | 0.3 |
| A7_PCA | V1 | reference | recent_payload82 | 0.1 | 0.75 | 0.82 | 0.65 |
| A7_Statistical | V1 | reference | recent_payload82 | 0.4 | 0.8 | 0.77 | 0.4 |
| A2_IsolationForest | V2 | A | recent_payload82 | 0.0 | 0.35 | 0.615 | 0.35 |
| A2_PCA | V2 | A | recent_payload82 | 0.0 | 0.05 | 0.665 | 0.05 |
| A2_Statistical | V2 | A | recent_payload82 | 0.0 | 0.35 | 0.57 | 0.35 |
| A7_IsolationForest | V2 | A | recent_payload82 | 0.1 | 0.35 | 0.625 | 0.25 |
| A7_PCA | V2 | A | recent_payload82 | 0.1 | 0.15 | 0.47 | 0.05 |
| A7_Statistical | V2 | A | recent_payload82 | 0.1 | 0.3 | 0.705 | 0.2 |
| A2_IsolationForest | V2 | B | recent_payload82 | 0.1 | 0.4 | 0.605 | 0.3 |
| A2_PCA | V2 | B | recent_payload82 | 0.0 | 0.0 | 0.63 | 0.0 |
| A2_Statistical | V2 | B | recent_payload82 | 0.0 | 0.4 | 0.57 | 0.4 |
| A7_IsolationForest | V2 | B | recent_payload82 | 0.1 | 0.3 | 0.705 | 0.2 |
| A7_PCA | V2 | B | recent_payload82 | 0.2 | 0.2 | 0.43 | 0.0 |
| A7_Statistical | V2 | B | recent_payload82 | 0.1 | 0.4 | 0.74 | 0.3 |
| A2_IsolationForest | V1 | reference | recent_upper82 | 0.6 | 0.4 | 0.375 | -0.2 |
| A2_PCA | V1 | reference | recent_upper82 | 0.1 | 0.0 | 0.32 | -0.1 |
| A2_Statistical | V1 | reference | recent_upper82 | 0.5 | 0.45 | 0.4 | -0.05 |
| A7_IsolationForest | V1 | reference | recent_upper82 | 0.9 | 1.0 | 0.685 | 0.1 |
| A7_PCA | V1 | reference | recent_upper82 | 0.7 | 0.85 | 0.67 | 0.15 |
| A7_Statistical | V1 | reference | recent_upper82 | 0.9 | 1.0 | 0.795 | 0.1 |
| A2_IsolationForest | V2 | A | recent_upper82 | 0.2 | 0.4 | 0.685 | 0.2 |
| A2_PCA | V2 | A | recent_upper82 | 0.0 | 0.2 | 0.715 | 0.2 |
| A2_Statistical | V2 | A | recent_upper82 | 0.2 | 0.35 | 0.6 | 0.15 |
| A7_IsolationForest | V2 | A | recent_upper82 | 0.1 | 0.9 | 0.935 | 0.8 |
| A7_PCA | V2 | A | recent_upper82 | 0.0 | 0.05 | 0.96 | 0.05 |
| A7_Statistical | V2 | A | recent_upper82 | 0.2 | 0.7 | 0.815 | 0.5 |
| A2_IsolationForest | V2 | B | recent_upper82 | 0.3 | 0.4 | 0.58 | 0.1 |
| A2_PCA | V2 | B | recent_upper82 | 0.0 | 0.0 | 0.35 | 0.0 |
| A2_Statistical | V2 | B | recent_upper82 | 0.2 | 0.35 | 0.555 | 0.15 |
| A7_IsolationForest | V2 | B | recent_upper82 | 0.1 | 0.95 | 0.95 | 0.85 |
| A7_PCA | V2 | B | recent_upper82 | 0.1 | 0.55 | 0.92 | 0.45 |
| A7_Statistical | V2 | B | recent_upper82 | 0.2 | 0.7 | 0.78 | 0.5 |
| A2_IsolationForest | V1 | reference | recent_lower82 | 0.2 | 0.15 | 0.46 | -0.05 |
| A2_PCA | V1 | reference | recent_lower82 | 0.1 | 0.05 | 0.35 | -0.05 |
| A2_Statistical | V1 | reference | recent_lower82 | 0.1 | 0.1 | 0.525 | 0.0 |
| A7_IsolationForest | V1 | reference | recent_lower82 | 0.1 | 0.55 | 0.74 | 0.45 |
| A7_PCA | V1 | reference | recent_lower82 | 0.3 | 0.75 | 0.71 | 0.45 |
| A7_Statistical | V1 | reference | recent_lower82 | 0.1 | 0.25 | 0.565 | 0.15 |
| A2_IsolationForest | V2 | A | recent_lower82 | 0.0 | 0.0 | 0.515 | 0.0 |
| A2_PCA | V2 | A | recent_lower82 | 0.1 | 0.05 | 0.52 | -0.05 |
| A2_Statistical | V2 | A | recent_lower82 | 0.0 | 0.0 | 0.48 | 0.0 |
| A7_IsolationForest | V2 | A | recent_lower82 | 0.0 | 0.1 | 0.82 | 0.1 |
| A7_PCA | V2 | A | recent_lower82 | 0.0 | 0.15 | 0.895 | 0.15 |
| A7_Statistical | V2 | A | recent_lower82 | 0.0 | 0.05 | 0.73 | 0.05 |
| A2_IsolationForest | V2 | B | recent_lower82 | 0.0 | 0.0 | 0.48 | 0.0 |
| A2_PCA | V2 | B | recent_lower82 | 0.1 | 0.05 | 0.415 | -0.05 |
| A2_Statistical | V2 | B | recent_lower82 | 0.0 | 0.0 | 0.475 | 0.0 |
| A7_IsolationForest | V2 | B | recent_lower82 | 0.0 | 0.05 | 0.78 | 0.05 |
| A7_PCA | V2 | B | recent_lower82 | 0.0 | 0.05 | 0.675 | 0.05 |
| A7_Statistical | V2 | B | recent_lower82 | 0.0 | 0.05 | 0.655 | 0.05 |

## 후속 검토 후보

탐색 순위는 최근 세 조건의 평균 TPR−FPR에서 정상 홀드아웃 FPR을 뺀 값이다. 95% 설명분산 규칙에서 전 성분이 남은 PCA는 순위에서 제외했다. 이 순위는 본 평가에 이미 사용한 하중 자료에 의존한다.

| model_id | strategy | holdout_FPR | worst_session_FPR | mean_gap | exploratory_balance |
|---|---|---|---|---|---|
| B5_IsolationForest | A | 0.135 | 0.65 | 0.383 | 0.248 |
| A7_IsolationForest | B | 0.13 | 0.65 | 0.367 | 0.237 |
| B3_IsolationForest | A | 0.125 | 0.65 | 0.317 | 0.192 |

A2는 전류·전압 6개만 사용하며 추가 센서 변동이 입력에 들어오지 않는다. A7은 가속도와 위치까지 포함한다. B3와 B5는 날짜에 민감한 ID1 범위 및 표준편차 제거 효과를 분리한다. C8과 C9는 각각 전압·전류 표준편차를 제거하며 B7은 ID1을 제외한 위치만 사용한다. 입력 수, 모델 용량, PC 배치 점수 시간과 학습 시간은 모델 색인에 있다.

V1을 운영 기준으로 보존하고 V2는 개발 후보로 관리한다. 독립 날짜에는 전원 직후와 일정 반복 이후의 정상 41g 자동 원복을 세션으로 나눠 먼저 확보하고, 그 날짜의 정상 및 세 하중 조건을 학습 전 고정 평가로 남긴다. 재현성 차이에는 시간·온도·파지·시작 자세·종료 판정이 함께 얽혀 있다. 현재 자료로 하중과 날짜 변화의 물리적 원인을 분리할 수 없다.

## 기존 V2와 동일한 평가 자료 비교

검증 150회와 홀드아웃 200회의 ID 및 순서가 기존과 일치한다. 이상·짝 정상·과거 참고 집단의 평가 ID도 일치한다. 표의 비율은 0~1이다. B는 기존과 같은 날짜별 100회 정원을 유지하며 후반 실험 묶음에서 신규 82회와 기존 18회를 사용했다.

| model_id | strategy | holdout_FPR_V2 | holdout_FPR_V21 | recent_payload82_TPR_V2 | recent_payload82_TPR_V21 | recent_upper82_TPR_V2 | recent_upper82_TPR_V21 | recent_lower82_TPR_V2 | recent_lower82_TPR_V21 |
|---|---|---|---|---|---|---|---|---|---|
| A2_IsolationForest | A | 0.12 | 0.1 | 0.4 | 0.35 | 0.4 | 0.4 | 0.0 | 0.0 |
| A2_IsolationForest | B | 0.095 | 0.105 | 0.25 | 0.4 | 0.4 | 0.4 | 0.0 | 0.0 |
| A7_IsolationForest | A | 0.115 | 0.155 | 0.35 | 0.35 | 0.85 | 0.9 | 0.1 | 0.1 |
| A7_IsolationForest | B | 0.135 | 0.13 | 0.25 | 0.3 | 0.75 | 0.95 | 0.1 | 0.05 |
| B3_PCA | A | 0.03 | 0.015 | 0.1 | 0.15 | 0.35 | 0.1 | 0.15 | 0.2 |
| B3_PCA | B | 0.035 | 0.055 | 0.15 | 0.2 | 0.85 | 0.45 | 0.15 | 0.2 |
| B5_PCA | A | 0.035 | 0.015 | 0.2 | 0.15 | 0.85 | 0.1 | 0.25 | 0.2 |
| B5_PCA | B | 0.045 | 0.03 | 0.15 | 0.15 | 0.75 | 0.1 | 0.2 | 0.0 |
| B7_Statistical | A | 0.075 | 0.065 | 0.25 | 0.0 | 1.0 | 0.35 | 0.1 | 0.1 |
| B7_Statistical | B | 0.075 | 0.065 | 0.2 | 0.05 | 1.0 | 0.5 | 0.1 | 0.1 |

정상 FPR이 낮아지고 세 하중 탐지율이 모두 유지되거나 오른 조합은 4개다. 절대 탐지율과 짝 정상 FPR은 아래 후보 및 전체 비교 CSV에서 확인한다.

| model_id | strategy | holdout_FPR_V2 | holdout_FPR_V21 | recent_payload82_TPR_V2 | recent_payload82_TPR_V21 | recent_upper82_TPR_V2 | recent_upper82_TPR_V21 | recent_lower82_TPR_V2 | recent_lower82_TPR_V21 |
|---|---|---|---|---|---|---|---|---|---|
| A2_PCA | A | 0.04 | 0.03 | 0.05 | 0.05 | 0.2 | 0.2 | 0.05 | 0.05 |
| C8_IsolationForest | A | 0.11 | 0.105 | 0.2 | 0.3 | 0.4 | 0.5 | 0.0 | 0.0 |
| C9_IsolationForest | A | 0.105 | 0.1 | 0.3 | 0.3 | 0.4 | 0.5 | 0.0 | 0.0 |
| C9_PCA | B | 0.045 | 0.035 | 0.05 | 0.05 | 0.15 | 0.2 | 0.0 | 0.0 |
