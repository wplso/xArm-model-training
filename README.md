# xArm 기준 모델 · 고정 평가 · 추가 원시 데이터

- [동결 기준 모델과 비교 자료](models/reference-models/README.md) · PC의 TabPFN S2, Raspberry Pi 4B 후보 로지스틱 S2, 공통 GRU ALL
- [모델별 성능](모델별%20성능.md) · 초기 정상 기반 모델부터 지도학습·사전학습·시계열 모델까지의 비교 결과
- [공통 370회 재생](models/reference-models/joint-replay/REPORT_KO.md) · PC 후보와 Raspberry Pi 4B 후보의 개별·OR 경보
- [신규 원시 데이터](원본%20수집%20데이터/볼트_추가이상_20261113/) · 정상 90회, D/E/F·폼 접촉·작업대 흔들림 각 30회 및 별도 폼 파일럿 2회
- [앞선 볼트 하중 모델 및 원시 데이터](models/bolt-known-load/README_KO.md)

가중치와 CSV 본문을 내려받으려면 Git LFS를 사용합니다.

```text
git lfs pull
```
