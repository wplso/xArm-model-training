# xArm V5.2 · 준비구간과 정상 전용 알고리즘 비교

V5.2는 정상·이상 회차에서 사전 정의한 준비구간을 제외하는 정책을 연구했습니다. 동일한 잔여 219회에서 V4 24특징 PCA-Q의 정상 오탐은 기존 33회, 재학습 35회였습니다. A/B/C 탐지는 기존 7·7·14회, 재학습 4·5·16회(각 20회)였습니다. V4 PCA-Q를 정상 기반 비교 기준으로 유지합니다.

- [V5.2 보고서와 모델](models/v5.2/README_KO.md)
- [V4 PCA Q·IF·PCA+IF 비교](models/v4-pca-if/REPORT_KO.md)
- [LOF·RBF One-Class SVM·kNN 비교](models/normal-only-benchmark/ALGORITHM_COMPARISON_REPORT_KO.md)
- [전체 버전과 후속 실험](https://github.com/wplso/xArm-model-training/blob/main/README.md)
- [이전 V5.1 코드·데이터 안내](https://github.com/wplso/xArm-model-training/blob/v5.1/README.md)

이 브랜치는 V5.1의 코드·모델과 [원시 데이터](원본%20수집%20데이터/)를 이어받습니다. 공개된 특징 CSV와 모델은 `git lfs pull`로 내려받습니다. 저장 모델의 점수는 배포본의 도구로 확인할 수 있습니다. 원시 자료부터 전체 실험을 재실행하는 원본 코드는 수집 당시의 프로젝트 파일 배치를 사용합니다.
