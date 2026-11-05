# V3·V3.1 재현용 보완 원시 세션

기존 공유 폴더에 없던 실제 사용 세션의 원본 CSV를 복사했다. 원본을 수정하거나 새 측정값을 생성하지 않았다.

- 역할: validation, 회차 목록 50개.
- measurements.csv: 전류·전압·전력과 상태, host_receive_monotonic_ns.
- adxl345_1.csv: 가속도와 호스트 수신 시각.
- single_queries.csv: 개별 관절 위치와 response_ns.
- cycles.csv: 포함 회차·라벨·No.3 시작/종료 시각. 목록에 없는 원시 대기·회차는 학습에서 제외한다.

V3와 V3.1의 정확한 역할은 상위 모델_학습_자료의 버전별 manifest를 따른다. 자동 원복 구간은 모델 입력에서 제외한다.
