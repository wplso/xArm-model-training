# 1031_No3_무부하_통합파일럿29

No.3 · 2026-10-31 KST · 목록 기준 3회.

개발·통합 검증 자료

| 파일 | 내용 |
|---|---|
| `measurements.csv` | INA226 전류(`current_A`), 전압(`voltage_V`), 전력(`power_W`), 수신 시각과 측정 상태. |
| `adxl345_1.csv` | ADXL345 1번의 XYZ 가속도(`x_g`, `y_g`, `z_g`), 원시값과 샘플 순번. |
| `single_queries.csv` | 서보별 실제 위치(`position_raw`), 응답 시각(`response_ns`), 회차 ID와 응답 상태. |
| `cycles.csv` | 공유 대상 회차의 라벨, 동작 구간, 기존 학습·검증·평가 역할. |

전류·가속도는 `host_receive_monotonic_ns`, 서보는 `response_ns`를 `action_begin_ns`~`action_end_ns`에 맞춰 사용한다. 시각 단위는 ns이며 같은 세션 안에서 비교한다. 동작 끝은 위치 기반 완료 판정 시각이다. 자동 원복 구간이 있는 회차는 `return_begin_ns`와 `return_end_ns`로 따로 기록했다. 원시 CSV의 대기 구간이나 제외 회차는 `cycles.csv`에 없는 ID 또는 시간 구간을 걸러낸다.
