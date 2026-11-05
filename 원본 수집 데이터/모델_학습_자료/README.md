# 바로 학습하거나 원시 데이터부터 재현하기

cycle_scores는 모델 출력이므로 학습 입력에 필요하지 않다.

1. 특징 CSV 경로: V3_CYCLE_FEATURES.csv 또는 V31_CYCLE_FEATURES.csv를 버전에 맞는 train_v3.py/train_v31.py에 전달한다. cohort=train 정상250회, validation 정상100회만 학습·임계값 결정에 사용한다.
2. 원시 경로: 버전별 CYCLE_MANIFEST.csv는 실제 cycle_id·session_id·No3_begin_ns·No3_end_ns·cohort와 공유 폴더 내 상대 source 경로를 제공한다. build_features.py가 각 source의 평면 파일 배치도 읽는다.
3. source 폴더마다 measurements.csv, adxl345_1.csv, single_queries.csv, cycles.csv가 있다. 별도 manifest.json은 이 공유 형식에서는 필요하지 않다. session_id는 cycles.csv로 검증한다.

기존 1101 정상450회는 이 모델의 고정 베이스 학습450회와 다르다. 버전별 manifest를 기준으로 사용하며 모든 공유 정상을 한꺼번에 학습에 섞지 않는다. V3.1의 추가120회는 평가용으로 유지한다. V3·V3.1 정상train/validation은 동일하고, V3.1은 V3를 대체한 확정 모델이 아니다.

원시 CSV를 수정하거나 라벨·질량·날짜를 모델 입력으로 사용하지 않는다. No.3 동작 구간만 사용하고 자동 원복은 제외한다. 전체 데이터의 기존 V2.1 역할과 V3·V3.1 역할을 혼동하지 않는다.
