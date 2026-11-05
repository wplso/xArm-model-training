# V3 입력 특징과 원시 컬럼

학습 입력은 `config/V3_TRAINING_CONFIG.json`의 명시적 목록으로만 선택합니다. `cycle_id`, 날짜·세션, 질량·이상 라벨, 수집 조건은 모델 입력이 아닙니다.

| 구성 | 입력 |
|---|---|
| R0 | 전체 전류 평균·RMS·표준편차·절댓값 P95, 전압 평균·표준편차 6개 |
| R1 | R0 + 가속도 x/y/z 표준편차·평균 제거 합성 가속도 RMS/피크 + ID1~6 위치 range/std, 총 23개 |
| V3-A | R1 + ID1 3~5초 IQR |
| V3-B | R1 + S3/S4/S5 전류 평균 |
| V3-C | R1 + ID1 IQR + S3/S4/S5 전류 평균 |
| V3-D | V3-C에서 ID1 전체 사이클 위치 range 제거 |
| V3-E | V3-D에서 ID1 전체 사이클 위치 std 추가 제거 |
| V3-F | R1 + ID1 3~5초 최대 인접 변화량 + S3/S4/S5 전류 평균 |
| V3-G | V3-C + ID1 최대 인접 변화량 |
| V3-C-S34 | V3-C에서 S5 전류 평균 제거 |
| V3-C-fall | V3-C + S5 전류 감소 시각. train/validation 각각 유효값 90% 이상일 때 사용 |

구간 기준 시각 0은 **No.3 명령 쓰기 완료 시각**입니다. S3=3~4초, S4=4~5초, S5=5~6.5초이며 반열린 구간 `[시작, 끝)`의 원시 표본을 사용합니다. 원시 신호를 모델 입력 계산용으로 보간하지 않습니다.

- ID1 IQR은 3~5초의 정확히 4개 위치 응답에서 선형 분위수 Q75−Q25로 계산합니다. 표본 간 최대 간격은 0.75초입니다.
- 구간 전류 평균의 최소 표본 수는 S3/S4 8개, S5 12개이며 최대 간격 0.25초·sequence 단절 없음이 조건입니다.
- 전류 감소 시각은 4.5~5.3초와 6.5~6.9초 전류 중앙값의 중간 임계값보다 연속 두 표본이 작아지는 첫 시각을 5.2~6.6초에서 찾습니다.
- 품질 조건을 만족하지 않은 추가 특징은 NaN이며 정상 train 중앙값으로만 대치합니다. 원시 품질 실패 자체를 정상으로 바꾸는 기능은 아닙니다.
- 전체 사이클 통계는 No.3 시작~종료 대리판정의 원시 표본이며 끝점을 포함합니다. 표준편차는 모집단 표준편차(`ddof=0`), 전류 P95는 절댓값의 nearest-rank 방식입니다.
- `ID4_end_position_deviation`은 정상 train 종료 위치 중앙값 대비 편차로 저장하지만 V3 학습 입력에는 포함하지 않았습니다.

## 원시 입력의 필수 컬럼

| 파일 | 컬럼 |
|---|---|
| 세션 `manifest.json` | 실제 수집 `session_id` |
| `sensors/measurements.csv` | `host_receive_monotonic_ns`, `current_A`, `voltage_V`, `valid`, `status`, `seq` |
| `sensors/adxl345_1.csv` | `host_receive_monotonic_ns`, `x_g`, `y_g`, `z_g` |
| `robot/single_queries.csv` | `cycle_id`, `phase`, `servo_id`, `response_ns`, `position_raw`, `status`, `value_valid` |

`phase=No3_A_to_B`인 위치 응답만 사용합니다. B→A 자동 원복 구간은 학습 사이클에서 제외합니다. 가속도 수신 시각은 FIFO 각 표본의 정밀한 실제 변환 시각을 뜻하지 않고, 위치 응답 시각도 서보 내부 샘플 시각을 뜻하지 않습니다.
