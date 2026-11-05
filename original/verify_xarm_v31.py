"""Audit completed V3.1 outputs and record the campaign handoff."""

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
from PIL import Image


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def prepend(path, section):
    raw = path.read_bytes()
    if section.encode("utf-8") in raw:
        return
    bom = b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""
    path.write_bytes(bom + section.encode("utf-8") + raw[len(bom):])


def main():
    project, out = (Path(v) for v in sys.argv[1:3])
    config = read(out/"TRAINING_CONFIG.json")
    frozen = read(out/"FROZEN_CANDIDATES.json")
    verification = read(out/"VERIFICATION.json")
    assert verification["source_and_prior_hashes_unchanged"]
    assert sha(out/"analysis_code/train_xarm_v31.py") == config["code_sha256"]
    for name, digest in config["input_sha256"].items():
        assert sha(Path(name)) == digest, name
    for name, digest in frozen["artifacts_sha256"].items():
        assert sha(out/name) == digest, name
    models = pd.read_csv(out/"MODEL_INDEX.csv", float_precision="round_trip")
    data = pd.read_csv(out/"DATA_MANIFEST.csv")
    assert len(data) == 788 and data.cycle_id.is_unique and len(models) == 54
    assert set(frozen["excluded_cycle_ids"]) == set(data.cycle_id)
    assert set(config["training_cycle_ids"]) == set(data.loc[data.cohort.eq("train"),"cycle_id"])
    assert set(config["validation_cycle_ids"]) == set(data.loc[data.cohort.eq("validation"),"cycle_id"])
    for row in models.itertuples(index=False):
        assert sha(Path(row.path)) == row.sha256, row.model_id
    for model in frozen["models"]:
        assert sha(Path(model["path"])) == model["sha256"], model["model_id"]
    scores = pd.read_csv(out/"metrics/CYCLE_SCORES.csv", float_precision="round_trip")
    assert len(scores) == 788*(33+21+54)
    assert scores.flagged.eq(scores.score.gt(scores.threshold)).all()
    figures = {}
    for path in sorted((out/"figures").glob("*.png")):
        with Image.open(path) as img:
            dimensions = img.size
            img.verify()
        assert min(dimensions) >= 480
        figures[path.name] = list(dimensions)
    assert len(figures) == 6
    dose = project/"hardware/No3_AC_dose_20261105"
    blocks = pd.read_csv(out/"verification/DOSE_AUDIT.csv")
    audit = []
    for block in blocks.itertuples(index=False):
        result = read(dose/f"result_{block.capture_stage}.json")
        raw = read(Path(result["folder"])/"manifest.json")
        condition = raw["conditions"]
        assert condition["base_mount"] == "rigid_fixed_operator_reported"
        assert condition["automatic_return"] is True and raw["pre_first_return_quiet_s"] == .3
        audit.append(dict(stage=block.capture_stage, first_pose=block.first_pose,
                          base_mount=condition["base_mount"], automatic_return=True,
                          before_return_quiet_s=.3, cycles=block.count))
    pd.DataFrame(audit).to_csv(out/"verification/DOSE_PROTOCOL_CHECK.csv", index=False, encoding="utf-8-sig")
    verification["saved_model_hashes_checked"] = len(models)
    verification["frozen_manifest_hashes_checked"] = len(frozen["models"])
    verification["all_recorded_decisions_match_strict_cutoff"] = True
    verification["decoded_plot_dimensions_px"] = figures
    verification["dose_raw_protocol_manifest_checks"] = len(audit)
    (out/"VERIFICATION.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")

    appendix = """## 해석 보충

- S4 평균은 단일 구간 후보 중 해석하기 쉬웠다. PCA에서 기존 23특징에 추가하면 A121은 3/20→6/20, C182는 5/20→8/20, 홀드아웃 FPR은 6/100→3/100이었다. 별도 정상은 1/20으로 같았다. 이 비교의 기준 모델은 R1 PCA다.
- S4 RMS의 학습 정상 순위 상관은 0.999였다. PCA에서 평균에 RMS를 더했을 때 A121과 C182가 모두 악화됐다. S4 표준편차와 P95는 PCA에서 A121을 각각 10/20, 12/20으로 개선했고 C182는 8/20으로 같았다. P95 조합은 정상 100회 5건, 별도 정상 20회 0건을 표시했으나 다른 정상 대조에서 최대 10%였다.
- V31-15 PCA는 V3-C-S34 PCA 대비 C182가 11→13/20, 홀드아웃 오탐이 13→6/100, 별도 정상 오탐이 0→1/20이다. A121은 모두 8/20, A193·C462는 모두 20/20이다. D·E·F는 유지됐으나 기존 A82·B82·C82는 모두 낮아졌다. V31-9 Statistical은 별도 정상 17/20 오탐으로 미세 하중 대조용이다.
- 8개 신규 블록의 원시 manifest에서 베이스 고정, B→A 자동 원복, 원복 전 300ms 유예를 확인했다. normal_a의 첫 회는 home, 나머지 신규 블록의 첫 회는 A_release였다. 질량별 물체의 형상과 파지 면이 동일했다는 검증은 없다. 종료 시각은 호스트 로그의 대리판정 시각이다.

"""
    report_path = out/"V31_REPORT_KO.md"
    contents = report_path.read_text(encoding="utf-8")
    if "## 해석 보충\n" in contents:
        first, tail = contents.split("## 해석 보충\n", 1)
        _, last = tail.split("## 후보 비교\n", 1)
        revised = first + appendix + "## 후보 비교\n" + last
        if revised != contents:
            report_path.write_text(revised, encoding="utf-8")
    else:
        contents = contents.replace("## 후보 비교\n", appendix+"## 후보 비교\n", 1)
        report_path.write_text(contents,encoding="utf-8")
    if "--handoff" not in sys.argv[3:]:
        return
    folder = "../../hardware/xarm_model_versions/v3_1_load_features/"
    links = (f"[V3.1 보고서]({folder}V31_REPORT_KO.md) · [독립 검증 계획]({folder}INDEPENDENT_VALIDATION_PLAN_KO.md)"
             f" · [고정 후보·임계값]({folder}FROZEN_CANDIDATES.json)")
    sections = {
        "WORK_LOG": ("V3.1 오프라인 실험 완료", "원시 788회와 저장 V3·V1-베이스고정 점수를 대조하고 V3.1 54개를 학습·재로딩했다. V31-15 PCA는 C182 13/20, A121 8/20, A193·C462 각 20/20, 정상 6/100·1/20이다. V3-C-S34 PCA 대비 기존 82g 세 조건은 하락했다. 신규 증량 120회는 평가·후보 검토 전용이다."),
        "DOCUMENT_INDEX": ("V3.1 하중 특징 개발 평가", "보고서, 독립 검증 절차, 동결 목록에서 시작한다. 결과 폴더의 DATA_MANIFEST.csv, MODEL_INDEX.csv, VERIFICATION.json에 분할·모델·재현 검증이 있다. 원본 증량 자료는 No3_AC_dose_20261105에 있다."),
        "DEVELOPMENT_PLAN": ("V3와 V3.1 다음 날짜 비교", "동일 베이스·시작 자세·자동 원복의 41g 정상과 A121·C182를 먼저 독립 검증한다. 큰 하중과 기존 이상도 별도로 재현한다. 저장 모델·특징 순서·전처리·임계값을 고정한다. 기존 788회와 제외 회차는 최종 독립 평가에 재사용하지 않는다."),
        "OPEN_QUESTIONS": ("V3.1의 하중·정상 세션 재현성", "C182 개선이 새 날짜에도 유지되는지, V3보다 기존 82g 탐지가 낮아지는지 확인해야 한다. ID4·ID5 고정창 위치 이동은 보조 관찰이다. 종료 정책과 2Hz 관측, 질량별 단일 블록 효과를 분리할 추가 근거가 필요하다."),
        "DECISIONS": ("V3 유지, V3.1 후보 동결", "사용자 결정으로 V3.1은 개발 후보로 유지한다. 다른 날짜에 동일 베이스·동일 시작 자세·동일 B→A 원복 조건의 정상·이상에서 두 버전의 모델과 임계값을 고정해 검증한 뒤 교체 여부를 판단한다."),
    }
    for name, (title, body) in sections.items():
        prepend(project/"docs/xarm"/f"{name}.md",f"## 2026-11-05 — {title}\n\n{body} {links}\n\n")


if __name__ == "__main__":
    main()
