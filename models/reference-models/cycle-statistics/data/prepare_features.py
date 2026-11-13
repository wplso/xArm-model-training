"""Audit saved bolt-mount cycles and cache one V4/phase feature row per cycle."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PureWindowsPath
import time

import numpy as np
import pandas as pd

HOST = 'host_receive_monotonic_ns'
AXES = ['x_g', 'y_g', 'z_g']
AXES_RAW = ['x_raw', 'y_raw', 'z_raw']
PHASES = dict(w0_3=(0, 3), w3_5=(3, 5), w5_6p5=(5, 6.5), w6p5_9p5=(6.5, 9.5), w9p5_12=(9.5, 12))
SIGNALS = {
    'E': ('current_mean', 'current_rms', 'current_std', 'current_abs_p95_nearest_rank', 'voltage_mean', 'voltage_std', 'voltage_min'),
    'P': tuple(f'ID{i}_median' for i in range(1, 7)),
    'A': ('centered_vector_rms', 'centered_vector_p95_linear', 'centered_vector_peak'),
}
RAW_FILES = {'ina': 'sensors/measurements.csv', 'adxl': 'sensors/adxl345_1.csv', 'pos': 'robot/single_queries.csv'}
TRAIN_MAIN = ('20261106_early_continuous', '20261111_normal240')
ABC06 = '20261106_ABC_validation'
ABC11 = '20261111_ABC_interleaved'
BC12 = '20261112_BC_independent'
CAL = '20261110_first_normal40'
SPLITS = {
    'primary': dict(train=(*TRAIN_MAIN, ABC06), development=(ABC11,), calibration=(CAL,)),
    'outer_hold_abc06': dict(train=(*TRAIN_MAIN, ABC11), development=(BC12,), calibration=(CAL,)),
    'outer_hold_abc11': dict(train=(*TRAIN_MAIN, ABC06), development=(BC12,), calibration=(CAL,)),
}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def csv(path, frame):
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def raw_source(project, recorded_path):
    parts = PureWindowsPath(recorded_path).parts
    root_name = 'voraus-ad-dataset-analysis'
    tail = parts[parts.index(root_name) + 1:]
    if not tail or tail[0] != 'hardware' or '..' in tail:
        raise ValueError(f'Unexpected recorded hardware path: {recorded_path}')
    return project.joinpath(*tail)


def load_existing(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def groups(v4):
    e = [name for name in v4 if name.startswith(('current_', 'voltage_'))]
    p = [name for name in v4 if name.startswith(('position_', 'ID1_'))]
    a = [name for name in v4 if name not in e + p]
    assert [len(e), len(p), len(a)] == [6, 13, 5] and len(set(v4)) == len(v4) == 24
    base = {'F_E': e, 'F_P': p, 'F_A': a}
    for combination in ('EP', 'EA', 'PA'):
        selected = {name for sensor in combination for name in base[f'F_{sensor}']}
        base[f'F_{combination}'] = [name for name in v4 if name in selected]
    phase = {key: [f'{window}__{key}__{name}' for window in PHASES for name in names] for key, names in SIGNALS.items()}
    return {**base, 'F_ALL24': v4, 'G0': v4, 'G1': v4 + phase['E'], 'G2': v4 + phase['P'], 'G3': v4 + phase['A'],
            'G4': v4 + phase['E'] + phase['P'] + phase['A']}


def matched_labels(row):
    if row.label != 'normal':
        return ''
    if row.pair_condition and not pd.isna(row.pair_condition):
        return row.pair_condition
    if row.leakage_group_id == ABC06:
        return {'preA': 'A121', 'preB': 'B82', 'preC': 'C182'}.get(row.capture_stage, '')
    if row.leakage_group_id == BC12:
        return 'B82|C182'
    return {'No3_bolt_DF_interleaved_20261113': 'D|F', 'No3_bolt_foam_contact30_20261113': 'foam_contact',
            'No3_bolt_station_disturbance30_20261113': 'station_disturbance'}.get(row.campaign_id, '')


def load_manifest(source):
    manifest = pd.read_csv(source, dtype={'No3_begin_ns': 'int64', 'No3_end_ns': 'int64'})
    assert len(manifest) == 1519 and manifest.cycle_id.is_unique and manifest.sensor_mount.eq('bolt_and_nut').all()
    assert manifest.quality_valid.all() and (manifest.No3_begin_ns < manifest.No3_end_ns).all()
    assert manifest.groupby(['group_id', 'label']).size().loc['20261113_final_anomaly_panel'].to_dict() == {
        'D': 30, 'E': 30, 'F': 30, 'foam_contact': 30, 'normal': 90, 'station_disturbance': 30}
    corrected = manifest.loc[manifest.label.isin(['D', 'F'])].groupby('capture_stage').label.unique().map(tuple).to_dict()
    assert corrected == {'D1': ('F',), 'D2': ('D',), 'F1': ('D',), 'F2': ('F',)}
    manifest = manifest.rename(columns={'group_id': 'previous_forecasting_group_id', 'role': 'previous_forecasting_role'})
    manifest['leakage_group_id'] = manifest.previous_forecasting_group_id.replace({'20261106_noon_normal30': ABC06})
    manifest['capture_id'] = manifest.session_id
    manifest['acquisition_session_id'] = manifest.session_id
    manifest['logical_collection_id'] = manifest.campaign_id + '/' + manifest.capture_stage
    manifest['start_pose_raw'] = manifest.start_pose
    manifest['start_pose'] = manifest.start_pose.replace({'A_release_verified_by_before_No3_target_gate': 'A_release'})
    manifest['matched_normal_set'] = [matched_labels(row) for row in manifest.itertuples()]
    manifest['manual_intervention_protocol'] = manifest.label.map({'E': 'manual object reset between cycles',
        'foam_contact': 'manual obstacle and object reset between cycles', 'station_disturbance': 'operator drop during cycle'}).fillna('')
    after_powered_rest = manifest.capture_stage.eq('powered_rest') & manifest.block_local_cycle.gt(20)
    manifest['restart_context'] = np.select([
        manifest.capture_stage.eq('after_powercycle'),
        after_powered_rest,
        manifest.capture_stage.eq('post') & manifest.campaign_id.eq('No3_41g_switch_off_rest_20261111'),
        manifest.capture_stage.isin(['N02', 'N03', 'N04']),
        manifest.capture_stage.eq('resumed13')],
        ['powercycle_50min', 'powered_rest_study', 'board_switch_off_5min', 'powered_rest_5min', 'collection_resumed'],
        default='other_or_unverified')
    manifest['post_rest_local_cycle'] = manifest.block_local_cycle.where(manifest.restart_context.ne('other_or_unverified'))
    manifest.loc[after_powered_rest,'post_rest_local_cycle'] -= 20
    return manifest


def split_manifest(manifest):
    outputs = []
    for split_id, spec in SPLITS.items():
        role_map = {group: role for role, groups in spec.items() for group in groups}
        assert len(role_map) == sum(map(len, spec.values()))
        table = manifest.assign(split_id=split_id, role=manifest.leakage_group_id.map(role_map).fillna('evaluation'))
        assert table.groupby('acquisition_session_id').role.nunique().max() == 1
        assert table.groupby('campaign_id').role.nunique().max() == 1
        assert table.groupby('leakage_group_id').role.nunique().max() == 1
        assert table.loc[table.role.eq('calibration'),'label'].eq('normal').all()
        outputs.append(table)
    return pd.concat(outputs, ignore_index=True)


def get_sources(folder):
    ina = pd.read_csv(folder / RAW_FILES['ina'], usecols=[HOST, 'current_A', 'voltage_V', 'valid', 'status'], float_precision='round_trip')
    ina = ina.loc[ina.valid.eq(1) & ina.status.eq('OK')].reset_index(drop=True)
    adxl = pd.read_csv(folder / RAW_FILES['adxl'], usecols=[HOST, *AXES, *AXES_RAW], float_precision='round_trip')
    pos = pd.read_csv(folder / RAW_FILES['pos'], usecols=['cycle_id','phase','status','value_valid','response_ns','servo_id','position_raw'],
                      dtype={'response_ns': 'Int64'}, float_precision='round_trip')
    pos = pos.loc[pos.phase.eq('No3_A_to_B') & pos.status.eq('complete') & pos.value_valid.eq(1)].sort_values('response_ns')
    assert ina[HOST].is_monotonic_increasing and adxl[HOST].is_monotonic_increasing
    return ina, adxl, dict(tuple(pos.groupby('cycle_id', sort=False)))


def segment(table, times, lo, hi, inclusive):
    begin = np.searchsorted(times, lo, side='left')
    end = np.searchsorted(times, hi, side='right' if inclusive else 'left')
    return table.iloc[begin:end]


def get_cycle(ina, adxl, pos_groups, i_clock, a_clock, cycle):
    lo, hi = cycle.No3_begin_ns, cycle.No3_end_ns
    pos = pos_groups[cycle.cycle_id]
    return (segment(ina, i_clock, lo, hi, True), segment(adxl, a_clock, lo, hi, True),
            segment(pos, pos.response_ns.to_numpy(np.int64), lo, hi, True))


def phase_values(i, a, p, start, end):
    values, counts = {}, {}
    i_clock = i[HOST].to_numpy(np.int64)
    a_clock = a[HOST].to_numpy(np.int64)
    p_clock = p.response_ns.to_numpy(np.int64)
    for phase, (left, right) in PHASES.items():
        lo, hi = start + round(left * 1e9), start + round(right * 1e9)
        assert lo < hi
        count_prefix = f'{phase}_n_'
        dt = right-left
        if hi > end:
            continue
        v = segment(i, i_clock, lo, hi, False)[['current_A','voltage_V']].to_numpy(float)
        v = v[np.isfinite(v).all(axis=1)]
        counts[count_prefix+'E'] = len(v)
        if len(v) >= math.ceil(8 * dt):
            current, voltage = v.T
            x = np.sort(np.abs(current))
            arr = [current.mean(), np.sqrt(np.mean(current**2)), current.std(ddof=0), x[math.ceil(.95*len(x))-1],
                   voltage.mean(), voltage.std(ddof=0), voltage.min()]
            values.update({f'{phase}__E__{name}': value for name, value in zip(SIGNALS['E'],arr)})
        position = segment(p, p_clock, lo, hi, False)
        for joint, name in enumerate(SIGNALS['P'], 1):
            samples = position.loc[position.servo_id.eq(joint),'position_raw'].dropna().to_numpy(float)
            counts[count_prefix+f'ID{joint}'] = len(samples)
            if len(samples) >= max(2, math.floor(1.5 * dt)):
                values[f'{phase}__P__{name}'] = np.median(samples)
        xyz = segment(a, a_clock, lo, hi, False)[AXES].to_numpy(float)
        xyz = xyz[np.isfinite(xyz).all(axis=1)]
        counts[count_prefix+'A'] = len(xyz)
        if len(xyz) >= math.ceil(80 * dt):
            norm = np.linalg.norm(xyz - xyz.mean(axis=0), axis=1)
            arr = [np.sqrt(np.mean(norm**2)), np.quantile(norm, .95, method='linear'), norm.max()]
            values.update({f'{phase}__A__{name}': value for name, value in zip(SIGNALS['A'],arr)})
    return values, counts


def extract(manifest, project, v4):
    hardware = project/'hardware'
    v3 = load_existing('pc_cycle_v3_feature_reuse', hardware/'xarm_model_versions/v3_phase_features/analysis_code/train_xarm_v3.py')
    pilot = load_existing('pc_cycle_pilot_describe_reuse', hardware/'normal_main_recheck_v2_20261031/pc/pilot_analysis.py')
    rows, qualities, perf, bench = [], [], [], {}
    for folder, part in manifest.groupby('source', sort=False):
        begin_read = time.perf_counter()
        ina, adxl, pos_groups = get_sources(raw_source(project, folder))
        read_ms = (time.perf_counter()-begin_read)*1000
        i_clock = ina[HOST].to_numpy(np.int64)
        a_clock = adxl[HOST].to_numpy(np.int64)
        for cycle in part.itertuples():
            start = time.perf_counter()
            i,a,p = get_cycle(ina, adxl, pos_groups, i_clock, a_clock, cycle)
            values = v3.full_features(i,a,p,pilot.describe)
            p1 = p.loc[p.servo_id.eq(1)]
            t = (p1.response_ns.to_numpy(np.int64)-cycle.No3_begin_ns)/1e9
            s = (t>=3)&(t<5)
            grip = p1.position_raw.to_numpy(float)[s]
            ok = len(grip)==4 and np.diff(t[s]).max()<=.75 and np.all(np.diff(t[s])>0)
            values['ID1_3_5s_IQR'] = float(np.diff(np.quantile(grip,[.25,.75],method='linear'))[0]) if ok else np.nan
            legacy_ms = (time.perf_counter()-start)*1000
            features, counts = phase_values(i, a, p, cycle.No3_begin_ns, cycle.No3_end_ns)
            extended_ms = (time.perf_counter()-start)*1000
            rows.append(dict(cycle_id=cycle.cycle_id, **{name:values.get(name,np.nan) for name in v4}, **features))
            rail = int(a[AXES_RAW].isin([-512,511]).any(axis=1).sum())
            qualities.append(dict(cycle_id=cycle.cycle_id, INA_n=len(i), ADXL_n=len(a), **{f'ID{j}_n': int(p.servo_id.eq(j).sum()) for j in range(1,7)},
                                  adxl_rail_count=rail, saturated=rail>0, **counts))
            perf.append(dict(cycle_id=cycle.cycle_id, no_io_legacy_ms=legacy_ms, no_io_extended_ms=extended_ms,
                             raw_csv_read_per_capture_ms=read_ms, capture_cycles=len(part), amortized_io_ms=read_ms/len(part)))
        key = part.leakage_group_id.iat[0]
        if key not in bench:
            bench[key] = dict(source=folder, read_csv_ms=read_ms, cycle_ids=part.cycle_id.tolist())
    return pd.DataFrame(rows), pd.DataFrame(qualities), pd.DataFrame(perf), bench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    project, out = args.project.resolve(), args.output.resolve()
    data_dir = out/'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    h = project/'hardware'
    forecast = h/'xarm_forecasting/forecast_multitarget_v1/DATA_MANIFEST.csv'
    reference = h/'xarm_model_versions/v4_bolt_fixed/FEATURE_DEFINITIONS.json'
    fixture_path = h/'xarm_model_versions/v4_bolt_fixed/metrics/CYCLE_FEATURES.csv'
    prior_hash_path = h/'xarm_forecasting/forecast_v1/verification/SOURCE_FILE_HASHES.csv'
    sources = {path.as_posix():sha(path) for path in (forecast,reference,fixture_path,prior_hash_path,
        h/'xarm_model_versions/v3_phase_features/analysis_code/train_xarm_v3.py',
        h/'normal_main_recheck_v2_20261031/pc/pilot_analysis.py')}
    manifest = load_manifest(forecast)
    assert manifest.source.nunique() == manifest.session_id.nunique() == 62
    assert manifest.previous_forecasting_group_id.nunique() == 10 and manifest.leakage_group_id.nunique() == 9
    old = read_json(reference)
    legacy = old['ordered_features']['V4_PCA']
    feature_groups = groups(legacy)
    prior_hashes = pd.read_csv(prior_hash_path).set_index('source')
    hashes = []
    for folder in manifest.source.unique():
        for raw in RAW_FILES.values():
            path = raw_source(project, folder)/raw
            digest = sha(path)
            expected = prior_hashes.loc[folder,raw.replace('/','_').replace('.','_')+'_sha256']
            assert digest == expected, path
            hashes.append(dict(source=folder, relative_file=raw, sha256=digest, matches_previous_audit=True))
    csv(data_dir/'RAW_SOURCE_HASHES.csv',pd.DataFrame(hashes))
    save_json(out/'FEATURE_GROUPS.json', feature_groups)
    save_json(out/'FEATURE_DEFINITIONS.json', dict(legacy_features=legacy, legacy_definitions={key:old['definitions'][key] for key in legacy},
        legacy_ID1=old['ID1'], legacy_source=str(reference), legacy_window='[No3 send completed, No3 completion proxy], inclusive',
        phase_timebase='PC host receive (individual position reply time) relative to No.3 command send completed',
        phase_windows_s=PHASES, phase_interval='left inclusive, right exclusive; complete within No3 only', phase_names=SIGNALS,
        phase_minimum=dict(E='ceil(8 * interval_s) finite current/voltage pairs', P='max(2,floor(1.5 * interval_s)) positions per ID',
                           A='ceil(80 * interval_s) finite XYZ samples'),
        phase_units=dict(E='current A, voltage V', P='raw response count', A='g'),
        phase_std_ddof=0, phase_current_P95='absolute current nearest rank, ceil(.95*n)',
        phase_vibration='XYZ minus axis mean, Euclidean norm then RMS / linear P95 / maximum',
        phase_missing='NaN if interval incomplete or minimum not satisfied; model training median imputation only'))
    initial_split = split_manifest(manifest)
    csv(data_dir/'PRESPECIFIED_SPLIT.csv',initial_split[['cycle_id','split_id','role','leakage_group_id']])
    started = time.perf_counter()
    features, quality, perf, bench = extract(manifest, project, legacy)
    time_extract = time.perf_counter()-started
    features = features.reindex(columns=['cycle_id',*feature_groups['G4']])
    fixture = pd.read_csv(fixture_path, usecols=['cycle_id',*legacy],float_precision='round_trip').set_index('cycle_id')[legacy]
    matching = features.set_index('cycle_id').loc[fixture.index,legacy]
    np.testing.assert_allclose(matching, fixture, equal_nan=True, atol=1e-10, rtol=1e-10)
    assert features.cycle_id.is_unique and len(features)==1519 and features[legacy].notna().all().all()
    manifest = manifest.merge(quality,on='cycle_id',validate='one_to_one')
    for key, names in feature_groups.items():
        manifest[f'{key}_complete'] = features[names].notna().all(axis=1).to_numpy()
    assert manifest.saturated.sum() > 0
    assert manifest.quality_valid.all() and manifest.F_ALL24_complete.all()
    csv(out/'DATA_USAGE_MANIFEST.csv',manifest)
    csv(out/'SPLIT_MANIFEST.csv',split_manifest(manifest))
    csv(data_dir/'FEATURES.csv',features)
    csv(data_dir/'EXTRACTION_PERFORMANCE.csv',perf)
    csv(data_dir/'FEATURE_MISSINGNESS.csv',features.drop(columns='cycle_id').isna().sum().rename_axis('feature').reset_index(name='missing_cycles'))
    normal = manifest.loc[manifest.label.eq('normal')].merge(features,on='cycle_id',validate='one_to_one')
    duration = []
    for group, part in normal.groupby('leakage_group_id'):
        for feature in legacy:
            for exposure in ['No3_proxy_duration_s', 'INA_n', 'ADXL_n', *[f'ID{j}_n' for j in range(1,7)]]:
                x,y = part[feature],part[exposure]
                rho = x.rank().corr(y.rank()) if x.nunique()>1 and y.nunique()>1 else np.nan
                duration.append(dict(leakage_group_id=group, normal_cycles=len(part), feature=feature, quality_exposure=exposure, spearman_rho=rho))
    csv(data_dir/'NORMAL_WINDOW_FEATURE_CORRELATIONS.csv',pd.DataFrame(duration))
    counts = manifest.groupby(['label','saturated']).size().to_dict()
    save_json(data_dir/'DATA_VERIFICATION.json',dict(source_hashes=sources, raw_files_matched_prior_audit=len(hashes), v4_replay_rows=len(matching),
        v4_max_abs_error=float(np.nanmax(np.abs(matching.to_numpy()-fixture.to_numpy()))), cycles=len(manifest),
        groups=manifest.leakage_group_id.nunique(), raw_captures=manifest.source.nunique(),
        rail_samples=int(manifest.adxl_rail_count.sum()), rail_cycles=int(manifest.saturated.sum()),
        phase_feature_count=len(feature_groups['G4'])-24, io_included_total_extraction_s=time_extract, capture_timing_examples=bench))
    print('Data ready', time_extract, len(manifest), features.shape, counts, flush=True)
    print(split_manifest(manifest).groupby(['split_id','role','label']).size().to_string(), flush=True)


if __name__ == '__main__':
    main()
