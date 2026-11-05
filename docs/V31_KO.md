# V3.1 코드 읽기

V3.1도 정상 train250회로만 모델·결측 대치·정규화를 적합하고, 정상 validation100회 점수의 95% 분위수(higher)를 임계값으로 사용한다. test 데이터로 특징·임계값을 사후 보정하지 않는다. 알고리즘 구현은 V3와 같은 `v3/detectors.py`다.

입력은 `V31-0`~`V31-17`의 18개 구성, 알고리즘은 Statistical·IsolationForest·PCA 세 가지여서 모델은 54개다. `original/train_xarm_v31.py`의 `configurations()`가 실제 구성 생성 코드이고, 저장된 목록은 `config/V31_TRAINING_CONFIG.json`에 있다.

기본 23개 특징에서 S3/S4/S5 전류 평균의 조합, S4 RMS·표준편차·절댓값 P95, S4−S3 전류차·S4/S3 비율, ID1 3~5초 IQR을 비교한다. V31-15/16/17은 특정 구성에서 ID1 전체 사이클 range를 제거한다.

- 구간은 V3와 같은 원시 표본·반열린 시간창을 사용한다.
- S4 std는 ddof=0, 절댓값 P95는 nearest-rank(`inverted_cdf`)다.
- 비율은 S3 평균 절댓값이 0.05A 이상일 때만 계산하며 나머지는 NaN이다.
- ID4·ID5 고정창 위치 분석은 부가 분석이며 모델 입력에는 넣지 않았다.
- `V31-0`은 기존 V3 R1과 같은 입력이어서 기준 점수 일치 검증에 사용했다.

`원본 수집 데이터/모델_학습_자료/V31_CYCLE_FEATURES.csv`로 `train_v31.py`를 실행하면 이전 개발의 788회에 같은 학습·평가 분할을 적용할 수 있다. 원시부터 계산할 때 `build_features.py --version V3.1`은 원본 `raw_extras()`를 호출한다.

V3.1은 V3를 자동 대체하지 않는다. 후보와 기존 임계값 참고는 `config/V31_FROZEN_REFERENCE.json`, 다음 날짜 검증 절차는 `V31_INDEPENDENT_VALIDATION_PLAN_KO.md`를 따른다. 임계값을 고정한 검증은 저장 모델·전처리 객체도 그대로 유지해야 하므로, 재학습 결과를 기존 고정 모델의 독립 검증으로 부르지 않는다.

파지 실패·이탈·위치 오류의 원복 없는 수집과, 운반·하중의 자동 원복 수집은 시작 자세·휴지 이력이 달라 별도로 해석한다. 개발 중 확인한 데이터의 점수를 최종 독립 성능이라고 주장하지 않는다.
