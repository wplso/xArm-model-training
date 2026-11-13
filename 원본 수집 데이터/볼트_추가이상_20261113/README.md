# 볼트 고정 xArm 실측 원시 데이터 — 2026년 11월 13일

베이스를 고정하고 ADXL345를 볼트와 너트로 장착한 xArm에서 수집한 No.3 평가 240회와 폼 접촉 파일럿 2회입니다. 평가의 정상 90회는 41g 물체를 운반하고 B→A로 자동 원복했습니다. 이상 유형은 아래와 같으며, 작업자가 각 블록의 실제 결과를 확인했습니다.

| 조건 | 회차 | 실제 결과 | 자동 원복 |
|---|---:|---|---|
| `normal` | 90 | 41g 운반 및 원복 성공 | 있음 |
| `D` | 30 | A 근처의 82g 물체를 처음부터 들어 올리지 못함 | 없음 |
| `E` | 30 | 82g 물체를 잡아 들어 올리다가 놓침 | 없음 |
| `F` | 30 | 82g 물체가 팔의 도달 범위 밖에 있어 접촉·운반 없음 | 없음 |
| `station_disturbance` | 30 | 작업대 충격 및 41g 운반·원복 확인 | 있음 |
| `foam_contact` | 30 | 우회전 중 폼 접촉 및 41g 운반 확인 | 없음 |

수집 블록 이름 `D1`·`F1`은 당시 계획을 보존합니다. 실제 관찰에 따른 평가 라벨은 `D1`에 F, `F1`에 D입니다. 학습·평가에는 `capture_stage` 대신 `label`을 사용하세요. 작업대 충격은 470g, 작업대 위 460mm, 목표 접촉 시점은 No.3 시작 후 10.5초였습니다. 실제 충격이나 폼 접촉 시각은 독립적으로 계측하지 않았습니다. `object_mass_g`는 시험 물체의 질량이며, D·E·F에서 그 질량이 동작 내내 운반됐음을 의미하지 않습니다.

| 파일 | 내용 |
|---|---|
| [`CYCLE_MANIFEST.csv`](CYCLE_MANIFEST.csv) | 평가 240회의 정정 라벨, 시작 자세, No.3·원복 경계 |
| [`PILOT_CYCLE_MANIFEST.csv`](PILOT_CYCLE_MANIFEST.csv) | 평가와 분리한 접촉 파일럿 2회. 첫 번째는 접촉만 별도 확인했고, 두 번째는 접촉과 운반을 확인했습니다. |
| [`SESSION_INDEX.csv`](SESSION_INDEX.csv) | 세션의 수집 날짜·블록·라벨·행 수와 원본 감사 및 백업 SHA-256 |
| [`FILE_MANIFEST.csv`](FILE_MANIFEST.csv) | 목록 자체를 제외한 배포 파일의 크기와 SHA-256 |
| `sessions/<session_id>/measurements.csv` | INA226 전류·전압·전력과 계측 상태 |
| `sessions/<session_id>/adxl345_1.csv` | ADXL345 X/Y/Z 가속도와 FIFO·수신 상태 |
| `sessions/<session_id>/single_queries.csv` | 6개 서보의 개별 위치 응답 |
| `sessions/<session_id>/cycles.csv` | 해당 세션만의 회차 메타데이터. `pilots/<session_id>/`도 같은 구조입니다. |

`source`는 해당 목록 파일이 있는 폴더 기준 상대 경로입니다. `No3_begin_ns`는 명령 송신 완료, `No3_end_ns`는 위치 기반 완료 대리판정의 PC 단조 시각(ns)입니다. 자동 원복이 있는 회차는 `return_begin_ns`·`return_end_ns`로 원복을 별도 구분합니다. 전류·가속도는 `host_receive_monotonic_ns`, 서보 위치는 `response_ns`로 같은 세션의 분석 구간과 정렬할 수 있습니다. ADXL 수신 시각은 FIFO 배치 단위입니다.

평가 16개 세션과 파일럿 2개 세션의 원시 CSV 54개는 원본·검증된 백업과 바이트 단위로 일치합니다. 데이터 파일은 Git LFS로 관리합니다. 파일럿은 평가 점수 집계에서 제외했습니다.
