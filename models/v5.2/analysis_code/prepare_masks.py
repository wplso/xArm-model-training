"""Audit existing captures and freeze V5.2 masks before any score computation."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


POLICY_ID = "PREP50_SHORT10_KEEP_PRENORMAL10_V1"
COLLECTIONS = {
    "nov06_main400": ("nov06_main400", 400, 50, False),
    "extra_normal17": ("nov06_extra_normal30", 30, 10, False),
    "extra_normal13": ("nov06_extra_normal30", 30, 10, False),
    "preA": ("nov06_preA", 30, 10, False),
    "preB": ("nov06_preB", 10, 0, True),
    "preC": ("nov06_preC", 10, 0, True),
    "A": ("nov06_A", 30, 10, False),
    "B": ("nov06_B", 30, 10, False),
    "C": ("nov06_C", 30, 10, False),
    "nov10_normal40": ("nov10_normal40", 40, 10, False),
    "nov10_restart40": ("nov10_restart40", 40, 10, False),
}
REQUIRED = """cycle_id source_path mount_id logical_collection_id acquisition_session_id
leakage_group_id original_cycle_order original_role label anomaly_condition quality_eligible
is_pre_anomaly_normal10 preparation_rule preparation_excluded post_preparation_eligible
exclusion_reason operator_confirmation_source history_uncertainty""".split()
EXTRA = """calendar_date capture_uuid original_capture_cycle_index recorded_cycle_index cohort
condition pair_condition role eval_session No3_begin_ns No3_end_ns start_pose automatic_return
payload_mass_g frame_added_mass_g frame_site""".split()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fields=None):
    with path.open("x", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def capture_evidence(capture):
    manifest = read_json(capture / "manifest.json")
    summary = read_json(capture / "summary.json")
    sends, aborts = {}, {}
    with (capture / "events.jsonl").open(encoding="utf-8-sig") as events:
        for line in events:
            event = json.loads(line)
            index = event.get("cycle_index")
            if event["kind"] == "motion_write_completed" and event.get("label") == "No3_once":
                if index in sends:
                    raise ValueError(f"Duplicate No.3 command in {capture}, index {index}")
                sends[index] = event["host_monotonic_ns"]
            elif event["kind"] == "aborted":
                aborts[index] = event["error"]
    return manifest, summary, sends, aborts


def read_attempts(project, old_rows):
    grouped = defaultdict(list)
    for row in old_rows:
        if row["quality_valid"] != "True" or row["sensor_mount"] != "bolt_and_nut_user_reported":
            raise ValueError(f"Unexpected V5.1 input status for {row['cycle_id']}")
        grouped[Path(row["source"])].append(row)
    attempts = defaultdict(list)
    capture_audit = []
    for capture, rows in grouped.items():
        manifest, summary, sends, aborts = capture_evidence(capture)
        logical_id = COLLECTIONS[rows[0]["eval_session"]][0]
        sid = manifest["session_id"]
        if {row["session_id"] for row in rows} != {sid}:
            raise ValueError(f"Capture/session mismatch: {capture}")
        completed = {int(item["cycle_index"]): item for item in summary["completed_cycles"]}
        if set(completed) != {int(row["cycle_index"]) for row in rows}:
            raise ValueError(f"V5.1 and raw completed cycles differ: {capture}")
        for row in rows:
            index = int(row["cycle_index"])
            item = completed[index]
            begin = int(row["No3_begin_ns"])
            if ((item["No3_write_completed_ns"], item["No3_confirmed_ns"]) != (begin, int(row["No3_end_ns"]))
                    or not 0 <= begin - sends[index] <= 10_000_000):
                raise ValueError(f"Cycle boundary mismatch: {row['cycle_id']}")
            attempts[logical_id].append((row, True, ""))
        for index, reason in aborts.items():
            cycle_id = f"{sid}_{index}" if index in sends else ""
            capture_audit.append(dict(
                logical_collection_id=logical_id, capture_uuid=sid, source_path=capture,
                recorded_cycle_index=index, no3_command_sent=index in sends,
                cycle_id=cycle_id, included_in_preparation_mask=index in sends,
                reason=reason, event_kind="aborted",
            ))
            if index in sends:
                row = dict(rows[-1], cycle_id=cycle_id, cycle_index=index,
                           capture_cycle_index=index - manifest["block_offset"],
                           No3_begin_ns=sends[index], No3_end_ns="", start_pose="",
                           operator_confirmation_source="",
                           normal_confirmation_source="No completed transport/return confirmation for this attempt")
                attempts[logical_id].append((row, False, reason))
        if len(sends) != len(rows) + len(sends.keys() & aborts.keys()):
            raise ValueError(f"Unaccounted physical commands: {capture}")

    zero = project / "hardware/No3_bolt_normal30_resume13_20261106_122054/runs/20261106_122113_resume13"
    manifest, summary, sends, aborts = capture_evidence(zero)
    if sends or summary["completed_count"] or list(aborts) != [18]:
        raise ValueError("Recorded zero-motion restart differs from expected start-pose stop")
    capture_audit.append(dict(
        logical_collection_id="nov06_extra_normal30", capture_uuid=manifest["session_id"],
        source_path=zero, recorded_cycle_index=18, no3_command_sent=False,
        cycle_id="", included_in_preparation_mask=False, reason=aborts[18], event_kind="aborted",
    ))
    expected_aborts = Counter([("nov06_main400", 400, False), ("nov06_extra_normal30", 18, True),
                              ("nov06_extra_normal30", 18, False)])
    if Counter((a["logical_collection_id"], a["recorded_cycle_index"], a["no3_command_sent"]) for a in capture_audit) != expected_aborts:
        raise ValueError("Unexpected failure audit counts")
    return attempts, capture_audit


def make_mask(attempts):
    mask = []
    for logical_id, values in attempts.items():
        chronological = sorted(values, key=lambda item: int(item[0]["No3_begin_ns"]))
        for order, (row, eligible, error) in enumerate(chronological, 1):
            _, _, exclude_count, exception = COLLECTIONS[row["eval_session"]]
            excluded = order <= exclude_count
            rule = "pre_anomaly_normal10_exception" if exception else f"exclude_first_{exclude_count}"
            reasons = []
            if not eligible:
                reasons.append("original_quality_exclusion " + error)
            if excluded:
                reasons.append("initial_preparation_window")
            condition = row["condition"]
            uncertainty = row["power_restart_evidence"] + " | Internal temperature not measured"
            if logical_id == "nov06_extra_normal30":
                uncertainty += " | MCU continuity across captures; position restoration and operator restart occurred"
            record = dict(
                cycle_id=row["cycle_id"], source_path=row["source"], mount_id="bolt_and_nut",
                logical_collection_id=logical_id, acquisition_session_id=row["acquisition_session_id"],
                leakage_group_id=row["leakage_group_id"], original_cycle_order=order,
                original_role=row["role"], label="normal" if eligible and condition == "normal" else
                "anomaly" if eligible else "unconfirmed_attempt",
                anomaly_condition=condition if condition in ("A", "B", "C") else "", quality_eligible=eligible,
                is_pre_anomaly_normal10=exception, preparation_rule=rule, preparation_excluded=excluded,
                post_preparation_eligible=eligible and not excluded, exclusion_reason=" | ".join(reasons),
                operator_confirmation_source=row["normal_confirmation_source"], history_uncertainty=uncertainty,
                calendar_date=row["calendar_date"], capture_uuid=row["capture_uuid"],
                original_capture_cycle_index=row["capture_cycle_index"], recorded_cycle_index=row["cycle_index"],
                cohort=row["cohort"], condition=condition, pair_condition=row["pair_condition"],
                role=row["role"], eval_session=row["eval_session"], No3_begin_ns=row["No3_begin_ns"],
                No3_end_ns=row["No3_end_ns"], start_pose=row["start_pose"], automatic_return=row["automatic_return"],
                payload_mass_g=row["payload_mass_g"], frame_added_mass_g=row["frame_added_mass_g"], frame_site=row["frame_site"],
            )
            mask.append(record)
    return sorted(mask, key=lambda row: (row["calendar_date"], int(row["No3_begin_ns"])))


def make_split(mask):
    split = []
    for row in mask:
        role = row["original_role"]
        evaluate = role in ("test", "evaluation_only")
        eligible, after = row["quality_eligible"], row["post_preparation_eligible"]
        split.append(dict(
            cycle_id=row["cycle_id"], original_role=role, role=role, quality_eligible=eligible,
            preparation_excluded=row["preparation_excluded"], post_preparation_eligible=after,
            fit_train=after and role == "train", fit_calibration=after and role == "validation",
            eval_e0=eligible and evaluate, eval_e1=after and evaluate, eval_e2=after and evaluate,
        ))
    return split


def retention_rows(mask, audit):
    by_collection = defaultdict(list)
    for row in mask:
        by_collection[row["logical_collection_id"]].append(row)
    result = []
    for logical_id, rows in by_collection.items():
        _, planned, exclude, exception = COLLECTIONS[rows[0]["eval_session"]]
        eligible = sum(r["quality_eligible"] for r in rows)
        retained = sum(r["post_preparation_eligible"] for r in rows)
        result.append(dict(
            logical_collection_id=logical_id, condition=rows[0]["condition"], planned_cycles=planned,
            commanded_cycles=len(rows), no_motion_stops=sum(a["logical_collection_id"] == logical_id and
                                                           not a["no3_command_sent"] for a in audit),
            quality_eligible=eligible, quality_excluded=len(rows) - eligible,
            preparation_excluded=sum(r["preparation_excluded"] for r in rows),
            preparation_excluded_eligible=eligible - retained,
            quality_and_preparation_overlap=sum(r["preparation_excluded"] and not r["quality_eligible"] for r in rows),
            post_preparation_eligible=retained, retention_rate=retained / eligible,
            n_full=eligible, n_excluded=eligible - retained, n_retained=retained,
            expected_initial_cut=exclude, pre_anomaly_normal10_exception=exception,
        ))
    return result


def validate(mask, split, old_rows):
    valid_ids = {r["cycle_id"] for r in mask if r["quality_eligible"]}
    assert len(mask) == len({r["cycle_id"] for r in mask}) == 650
    assert valid_ids == {r["cycle_id"] for r in old_rows} and len(valid_ids) == 649
    assert len({(r["logical_collection_id"], r["original_cycle_order"]) for r in mask}) == len(mask)
    assert not any(not r["quality_eligible"] and r["preparation_excluded"] for r in mask)
    assert all(a["eval_e1"] == a["eval_e2"] for a in split)
    counts = Counter((r["original_role"], r["condition"]) for r in mask if r["quality_eligible"])
    assert counts == {("train", "normal"): 200, ("validation", "normal"): 100, ("test", "normal"): 99,
                      ("evaluation_only", "normal"): 160, **{("evaluation_only", c): 30 for c in "ABC"}}
    for key, expected in (("fit_train", {"normal": 150}), ("fit_calibration", {"normal": 100}),
                          ("eval_e0", {"normal": 259, "A": 30, "B": 30, "C": 30}),
                          ("eval_e1", {"normal": 219, "A": 20, "B": 20, "C": 20})):
        assert Counter(r["condition"] for r, s in zip(mask, split, strict=True) if s[key]) == expected
    assert sum(r["preparation_excluded"] for r in mask if r["logical_collection_id"] == "nov06_main400") == 50
    assert sum(r["is_pre_anomaly_normal10"] for r in mask) == 20
    assert all(r["post_preparation_eligible"] for r in mask if r["is_pre_anomaly_normal10"])


def freeze(project, output):
    previous = project / "hardware/xarm_model_versions/v5_1_multisession_normal_study"
    source = previous / "DATA_USAGE_MANIFEST.csv"
    old_rows = read_csv(source)
    attempts, audit = read_attempts(project, old_rows)
    mask = make_mask(attempts)
    split = make_split(mask)
    validate(mask, split, old_rows)
    output.mkdir(parents=True, exist_ok=True)
    (output / "DATASET_AUDIT_KO.md").write_text(DATASET_AUDIT, encoding="utf-8")
    (output / "PREPARATION_POLICY_KO.md").write_text(POLICY, encoding="utf-8")
    write_csv(output / "PREPARATION_MASK.csv", mask, REQUIRED + EXTRA)
    write_csv(output / "SPLIT_MANIFEST.csv", split)
    write_csv(output / "COLLECTION_RETENTION_SUMMARY.csv", retention_rows(mask, audit))
    write_csv(output / "COLLECTION_ATTEMPT_AUDIT.csv", audit)
    counts = []
    for role in ("train", "validation", "test", "evaluation_only"):
        for condition in ("normal", "A", "B", "C"):
            rows = [r for r in mask if r["original_role"] == role and r["condition"] == condition]
            if rows:
                full = sum(r["quality_eligible"] for r in rows)
                retained = sum(r["post_preparation_eligible"] for r in rows)
                counts.append(dict(original_role=role, condition=condition, total_commanded=len(rows),
                                   quality_eligible=full, preparation_excluded=full - retained,
                                   post_preparation_eligible=retained))
    write_csv(output / "TRAIN_CALIBRATION_TEST_COUNTS.csv", counts)

    old_by_id = {r["cycle_id"]: r for r in old_rows}
    usage = []
    for row, assignment in zip(mask, split, strict=True):
        old = old_by_id.get(row["cycle_id"], {})
        usage.append(dict(
            row, source=row["source_path"], session_id=row["acquisition_session_id"], cycle_index=row["recorded_cycle_index"],
            quality_valid=row["quality_eligible"], physical_label=old.get("physical_label", "unconfirmed_attempt"),
            ina_csv_sha256=old.get("ina_csv_sha256", ""), adxl_csv_sha256=old.get("adxl_csv_sha256", ""),
            position_csv_sha256=old.get("position_csv_sha256", ""), raw_uart_sha256=old.get("raw_uart_sha256", ""),
            raw_hid_sha256=old.get("raw_hid_sha256", ""), fit_train=assignment["fit_train"],
            fit_calibration=assignment["fit_calibration"], eval_e0=assignment["eval_e0"],
            eval_e1=assignment["eval_e1"], eval_e2=assignment["eval_e2"],
        ))
    write_csv(output / "DATA_USAGE_MANIFEST.csv", usage)
    mask_sha = digest(output / "PREPARATION_MASK.csv")
    (output / "PREPARATION_MASK.sha256").write_text(f"{mask_sha}  PREPARATION_MASK.csv\n", encoding="ascii")
    receipt = dict(policy_id=POLICY_ID, source_manifest_sha256=digest(source),
                   preparation_mask_sha256=mask_sha, mask_rows=len(mask), eligible_rows=len(old_rows),
                   launched_quality_exclusions=1, no_motion_stops=2, verified_raw_capture_count=19,
                   checks=["cycle_id_and_chronological_order_unique", "raw_command_and_boundaries_replayed",
                           "existing_quality_eligible_ids_identical", "original_roles_retained", "e1_e2_masks_identical"])
    (output / "PREPARATION_AUDIT_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


DATASET_AUDIT = """# V5.2 데이터셋 감사

이번 본수집은 **2026-11-06 볼트·너트 고정 41g No.3 운반 및 B→A 자동 원복 399회**다. V4 품질 감사와 V5.1 데이터 사용 명세의 동일한 399개 Cycle ID를 원시 `summary.json` 및 `events.jsonl`의 No.3 실제 송신 완료·종료 대리판정 시각과 대조했다. 이 399회는 여덟 파일 블록에 기록된 한 연속 운전이다. 별도 450회 볼트 본수집은 해당 실험 입력 및 기록에서 확인되지 않았고, 이전 450회는 가속도 센서 테이프 장착 조건이다. V4의 정상 200/100/99 분할을 유지한다.

| 논리 수집 | 품질 통과 | 준비 제외 | 유지 |
|---|---:|---:|---:|
| 11월 6일 볼트 본수집 | 399 | 50 | 349 |
| 11월 6일 별도 정상 17+13 | 30 | 10 | 20 |
| A121 직전 정상 | 30 | 10 | 20 |
| B82·C182 직전 정상 | 각 10 | 0 | 각 10 |
| A121·B82·C182 | 각 30 | 각 10 | 각 20 |
| 11월 10일 최초 정상 및 재가동 정상 | 각 40 | 각 10 | 각 30 |

본수집 #400은 위치 시작 검사를 기록한 뒤 캠페인 시간 상한으로 중단했다. Stop 송신은 기록됐지만 해당 번호의 No.3 송신·시작 이벤트가 없다. 이 번호를 실제 수행 회차로 만들지 않았고, 계획 대비 차이는 `COLLECTION_ATTEMPT_AUDIT.csv`에 남겼다. 마지막 b08의 완료 49회는 기존 별도 품질 감사에 따라 유지한다.

별도 정상30의 첫 캡처에서 17회 완결 후 #18에 No.3을 송신했으나 위치 신선도 검사에서 중단됐다. 응답 6개가 모두 도착했으나 시각 차와 가장 오래된 값 기준을 넘었다. 첫 재개는 시작 자세 검사에서 멈춰 No.3을 보내지 않았다. 다음 재개로 13회를 실제 수행했다. 이 13회는 수집기 번호 18–30을 재사용하지만 캡처 UUID가 다르며, 실제 송신 순서는 실패를 포함하여 19–31이다. `recorded_cycle_index`와 `original_capture_cycle_index`를 보존하고 정책에는 `original_cycle_order`를 쓴다. 원래 실패 #18은 품질 부적격이며 준비 제외와 겹치지 않는다. 따라서 31개 실제 송신 중 30개가 품질을 통과했고 준비 10회를 분리한 뒤 20개가 남는다.

A 직전 정상30은 기존 별도 평가에서 참조된 같은 Cycle ID의 정상30이다. 중복 등록 없이 한 번만 적용한다. 11월 10일 최초40과 재가동40은 각 논리 단위다. 약 50분 로봇 OFF는 작업자 보고이며 캡처 경계 사이 MCU 기록은 이어졌다. 최초40의 첫 회는 home, 재가동40은 전회 A_release에서 시작했다. 날짜와 운전 맥락은 구분해 기록했다.

주 집단은 V5.1의 품질 통과 649회와 품질 감사용 실제 송신 실패 1회로 고정했다. 11월 11일 전원 ON/OFF 휴지와 전류 변화 진단, 이전 테이프 장착, 새 볼트 D/E/F는 주 분석에 포함하지 않았다. 휴지 자료는 이력과 구간이 일반 반복과 달라 이 정책의 안정화를 보증할 근거로 쓰지 않는다. 온도를 직접 측정하지 않았다. 모든 분석 자료는 이전에 열람한 개발 자료다.
"""

POLICY = """# 고정 준비구간 제외 정책

정책 ID는 `PREP50_SHORT10_KEEP_PRENORMAL10_V1`이다. 회차의 특징값·점수·경보를 사용하지 않고 수집 계획과 실제 송신 순서로 고정했다. `PREPARATION_MASK.sha256`은 학습 전 마스크의 SHA-256을 기록한다.

1. 기존 품질 기준을 적용한다. 미완료·오류 회차는 원시 기록을 보존하고 학습·평가에서는 제외한다. 실제 No.3을 보내지 않은 시작 실패는 별도 감사에만 남긴다.
2. B82·C182 직전의 확인된 정상10회는 `pre_anomaly_normal10_exception`을 적용한다. A121 직전 정상30은 소규모 규칙을 따른다.
3. 긴 연속 본수집은 첫50회만 분리한다. 블록 또는 Train/Calibration/Test 경계에서 다시 적용하지 않는다.
4. 독립적으로 계획된 소규모 정상·이상 수집은 첫10회를 분리한다. 중단 후 재개는 처음 계획한 논리 단위를 유지하고 실제 실패 송신도 원래 순서에 남긴다.
5. 준비 이후라는 표기는 분석 적용 범위를 가리킨다. 물리적 온도 상태와 안정화는 별도 검증 대상이다.

| 비교 | 적합과 점수 | 주 평가 |
|---|---|---|
| E0 | 기존 모델과 임계값 | 품질 통과 정상259·이상 각30 |
| E1 | E0와 같은 모델·점수·임계값 | 유지 정상219·이상 각20 |
| E2 | Train150 및 기존 Calibration100으로 새 적합·교정 | E1과 같은 Cycle ID |

준비구간 회차는 해당 제한 정책의 주 판정 대상에서 보류한다. 원래 실제 조건 라벨과 가능한 참고 점수는 유지한다. 구간 분리만 바꾼 E0→E1과 같은 회차에서 모델이 바뀐 E1→E2를 따로 해석한다. 실험 당시 이상 조건 전환이 알려져 있었으므로 운영 중 이상 발생 시점에 적용하는 규칙의 검증은 별도다.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    freeze(arguments.project, arguments.output)
