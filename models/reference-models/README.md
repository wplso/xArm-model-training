# xArm 동결 기준 모델과 비교 자료

2026-11-14에 확정한 개발 기준 모델과 핵심 비교 모델을 보존했습니다. 학습 가중치·정규화·임계값은 원래 파일을 사용했습니다.

| 자료 | 파일과 용도 |
|---|---|
| PC 회차 종료 | [TabPFN-v2 S2 P0](tabpfn/) · S2 특징 52개, 학습 문맥 809회, seed 7181, 4개 앙상블. `1−P(정상) > 0.19058864334053838` |
| Raspberry Pi 4B 회차 종료 후보 | [로지스틱 S2](tabpfn/baseline/models/primary_S2_B_LR_7181.joblib) · S2 52개와 학습 중앙값·스케일러. `1−P(정상) > 0.2907932588975627` |
| 공통 동작 중 감시 | [GRU F2·O3 ALL](forecast/runs/gru_base_F2_L2_O3_s7181/) · 과거 2초, 0.5초 뒤 예측, 0.2초 발행 간격. `ALL` 정규화 잔차 점수 `> 7.288925055167675` |
| 회차 통계 비교 | [CatBoost S2](tabpfn/baseline/models/primary_S2_B_CB_7181.joblib), [로지스틱 G2 독립 홀드아웃](g2-holdout/) |
| 시계열 비교 | [TCN F4](forecast/runs/tcn_large_F4_L2_O3_s7181/)는 GRU 잔차 경보의 대조, [XGBoost F7](forecast/runs/xgb_base_F7_L2_O3_s7181/)과 [HistGradientBoosting F2](hgb/)는 예측값 표시 후보 |

- [동결 설정과 입력 순서](FROZEN_MODELS.json), [공개 파일 SHA256](ARTIFACT_SHA256.json)
- [회차 특징 행렬](cycle-statistics/data/FEATURES.csv), [회차 모델 분할](tabpfn/SPLIT_MANIFEST.csv), [시계열 모델 분할](forecast/SPLIT_MANIFEST.csv). 공개용 분할에는 회차 ID, 조건, 학습·교정·평가 역할과 시간 경계가 들어 있습니다. 작업 PC의 절대 경로는 제거했습니다.
- [공통 370회 재생 결과](joint-replay/REPORT_KO.md), [회차별 점수와 경보](joint-replay/results/CYCLE_ALERTS.csv), [결합 판정 차이](joint-replay/results/OR_DISAGREEMENT_CYCLES.csv). 보존된 결과에서 OR는 같은 회차에 두 경보 중 하나라도 발생하면 참입니다. 기존 판정과 예측값은 로컬 원본 재생에서 일치했습니다. 전체 실험 재생에는 원본 프로젝트의 별도 정렬 캐시와 로그가 필요합니다.
- [모델별 성능 및 한계](https://github.com/wplso/xArm-model-training/blob/main/%EB%AA%A8%EB%8D%B8%EB%B3%84%20%EC%84%B1%EB%8A%A5.md)

## 파일 받기

브랜치를 받은 뒤 `git lfs pull`로 가중치와 CSV 본문을 내려받습니다. [고정 설정](FROZEN_MODELS.json)에 원래 실험에서 사용한 임계값과 파일별 SHA256이 기록되어 있습니다. 정상 분할의 학습·교정·평가 역할을 유지해야 기존 지표와 비교할 수 있습니다.

TabPFN의 외부 사전학습 가중치는 [Prior Labs의 고정 커밋에서 다운로드](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/a7e9a4696dd0efbd0e64a705645e0d137ca2de72/tabpfn-v2-classifier-v2_default.ckpt)해 `tabpfn/checkpoint/tabpfn-v2-classifier-v2_default.ckpt`에 둡니다. SHA256은 `cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49`입니다. [TabPFN 소스 라이선스와 별도 가중치 이용조건](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/blob/a7e9a4696dd0efbd0e64a705645e0d137ca2de72/LICENSE.txt)

Built with PriorLabs-TabPFN.
