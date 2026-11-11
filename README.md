# xArm 볼트 고정 하중 모델 · 독립 평가와 원시 데이터

[모델·코드·평가 목록](models/bolt-known-load/README_KO.md) · [추가 원시 데이터](원본%20수집%20데이터/볼트_하중_보완원시/) · [전체 버전 성능](https://github.com/wplso/xArm-model-training/blob/main/README.md)

No.3의 41g 운반·자동 원복과 A121·B82·C182 조건에서 학습하거나 평가한 4클래스 로지스틱, 이진 로지스틱, Random Forest 연구와 별도 캠페인 결과입니다. 세 신규 캠페인에서 24개 세션, 510회의 검증된 원시 시계열을 추가했습니다.

| 독립 평가 | 핵심 결과 |
|---|---|
| [11월 11일 A/B/C 교차 블록](models/bolt-known-load/abc-interleaved/frozen_evaluation/REPORT_KO.md) | V4 PCA-Q·몬스터겔 로지스틱·볼트 로지스틱의 정상 오탐은 19·2·14/90회, 하중 탐지는 44·50·59/90회였습니다. |
| [11월 12일 B/C 고정 임계값](models/bolt-known-load/bc-independent/REPORT_KO.md) | 동일 모델에서 0.821445→0.40으로 정상 오탐 3→6/30회, B82 탐지 23→26/30회, C182 탐지 23→28/30회였습니다. |

0.40은 후보 임계값이며 모델의 저장 임계값은 0.8214447226813868입니다. 새 B/C 캠페인 90회는 평가 전용입니다. 정상·학습·교정·평가 회차는 각 캠페인의 역할 목록을 따릅니다. CSV와 모델은 `git lfs pull`로 내려받습니다.

이 브랜치는 [V5.2](https://github.com/wplso/xArm-model-training/tree/v5.2)의 코드와 데이터를 이어받습니다.
