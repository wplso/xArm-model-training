"""Matched, group-aware evaluation of offline No.3 forecast predictions."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import Data, TARGETS, TARGET_GROUPS, UNITS


ROOT = Path(__file__).resolve().parent
NS = 1_000_000_000
STRIDE_NS = 200_000_000
NAMES = list(TARGETS)
GROUPS = {key: [NAMES.index(col) if isinstance(col, str) else col for col in cols]
          for key, cols in TARGET_GROUPS.items()}
GROUP_NAME = {col: group for group, cols in GROUPS.items() for col in cols}
NATIVE_UNITS = [UNITS[name] for name in NAMES] if isinstance(UNITS, dict) else list(UNITS)
TARGET_FLOOR = np.array([1e-4, 1e-4, *([.1] * 6), *([1e-5] * 6)])
PHASES = ['startup', 'grip_lift_transfer', 'place_return', 'late']
BASE_CONFIG = {'L': 2., 'horizons': [.5], 'w': .2, 'seed': 7181, 'group': 'F4', 'output': 'O3'}


def write_csv(rows, name, root=ROOT):
    target = Path(root) / 'metrics' / f'{name}.csv'
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix('.csv.tmp')
    (rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)).to_csv(temp, index=False, encoding='utf-8-sig')
    temp.replace(target)


def verify_forecast_predictions(spec, meta, pack, predictions):
    if np.isinf(predictions).any():
        raise ValueError(f'{spec["id"]}: infinite predictions')
    assert np.array_equal(pack['valid_mask'], np.isfinite(pack['y']))
    issue, begin, end = [meta[c].to_numpy(np.int64) for c in ('issue_ns','No3_begin_ns','No3_end_ns')]
    if not np.all((issue>begin) & (issue<end)):
        raise ValueError('A forecast is issued outside its cycle')
    nominal = pack['nominal_target_ns']
    if not np.all(nominal>issue[:,None]):
        raise ValueError('Forecast target is not in the future')
    target_ok = pack['valid_mask']
    if not np.all(~target_ok | ((pack['sample_ns'] >= nominal[:,:,None]) & (pack['target_seq'] >= 0))):
        raise ValueError('A scored target sequence or clock is invalid')
    if np.any(meta.input_last_ina_available_ns.to_numpy(np.int64) > issue):
        raise ValueError('A forecast used an INA message newer than its issue time')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def run_specs(root=ROOT, model_ids=None):
    paths = sorted((Path(root) / 'runs').glob('*/RESULT.json'))
    requested = set(model_ids or [])
    specs = []
    for path in paths:
        if requested and path.parent.name not in requested:
            continue
        if not (path.parent/'PREDICTIONS_development.npy').is_file():
            continue
        report = read_json(path)
        spec = report.get('spec', report.get('config', report))
        if 'horizons' not in spec:
            raise ValueError(f'Forecast spec is missing: {path}')
        specs.append(dict(spec, id=path.parent.name, _directory=path.parent, _result=report))
    if requested and {s['id'] for s in specs} != requested:
        raise ValueError(f'Missing model specs: {requested.difference(s["id"] for s in specs)}')
    return specs


def archived_specs(root=ROOT):
    base = Path(root) / 'cache' / 'v1_predictions'
    expected = ('v1_e_f4_tree','v1_mt_f2_tree','v1_mt_f2_ridge','v1_mt_f2_gru16')
    if any(not (base/name).is_dir() for name in expected):
        raise FileNotFoundError('Run reproduce_v1.py before forecast evaluation')
    return [dict(BASE_CONFIG, id=name, family='v1_archive', group='F4' if '_e_f4_' in name else 'F2',
                 size='frozen', output='O0' if '_e_f4_' in name else 'O3', _directory=base/name)
            for name in expected]


def key_for(spec):
    return float(spec['L']), tuple(map(float, spec['horizons'])), float(spec.get('w', .2))


def model_tags(spec):
    history, hs, width = key_for(spec)
    return {'model_id': spec['id'], 'family': spec.get('family'), 'size': spec.get('size'),
            'input': spec.get('group'), 'output': spec.get('output'), 'L': history,
            'w': width, 'horizons': '|'.join(f'{h:g}' for h in hs), 'seed': spec.get('seed')}


def basic_specs(specs):
    layouts = {key_for(s) for s in specs} | {key_for(BASE_CONFIG)}
    return [dict(BASE_CONFIG, family=family, id=f'{family}_L{history:g}_h{hs_text}_w{width:g}',
                 L=history, horizons=hs, w=width, size='rule', group='causal_current' if family == 'persistence' else 'elapsed_s')
            for history, hs, width in sorted(layouts)
            for hs_text in ['-'.join(f'{h:g}' for h in hs)]
            for family in ('persistence', 'template')]


def prepare_aligned(data, spec, role):
    history, hs, width = key_for(spec)
    pack = data.targets(role, history, list(hs), w=width)
    meta = pack['meta'].reset_index(drop=True)
    assert meta.role.eq(role).all() and not meta.duplicated(['cycle_id', 'issue_ns']).any()
    expected = len(meta), len(hs), len(NAMES)
    assert pack['y'].shape == expected
    # V1 records reuse over the archive grid. Also expose reuse within this model's requested outputs.
    requested_duplicate = np.zeros(expected,dtype=bool)
    for i in range(len(hs)):
        for j in range(i):
            match = (pack['valid_mask'][:,i] & pack['valid_mask'][:,j] &
                     (pack['target_seq'][:,i] == pack['target_seq'][:,j]) &
                     (pack['sample_ns'][:,i] == pack['sample_ns'][:,j]))
            requested_duplicate[:,i] |= match
            requested_duplicate[:,j] |= match
    pack['target_duplicate_requested_h'] = requested_duplicate
    return pack, meta


def baseline_values(data, spec, role, meta):
    history, hs, width = key_for(spec)
    archive_path = data.root/'cache'/'v1_predictions'/f'v1_{spec["family"]}'/f'{role}.npy'
    if (history,hs,width) == key_for(BASE_CONFIG) and archive_path.is_file():
        return np.load(archive_path,mmap_mode='r',allow_pickle=False)
    if spec['family'] == 'persistence':
        values = data.persistence(role, history, w=width) if width != .2 else data.persistence(role, history)
        return np.broadcast_to(values[:, None, :], (len(meta), len(hs), len(NAMES)))
    # V1 template bins elapsed nominal target time; use only normal Train windows.
    fit_spec = dict(spec, L=.5)
    try:
        pack, fit_meta = prepare_aligned(data, fit_spec, 'train')
    except (KeyError, ValueError):
        pack, fit_meta = prepare_aligned(data, dict(spec, L=1.), 'train')
    assert fit_meta.label.eq('normal').all()
    predictions = np.full((len(meta), len(hs), len(NAMES)), np.nan, dtype=np.float32)
    for i, h in enumerate(hs):
        fit_time = (fit_meta.issue_ns-fit_meta.No3_begin_ns+round(h*NS)).to_numpy()/NS
        issue_time = (meta.issue_ns-meta.No3_begin_ns+round(h*NS)).to_numpy()/NS
        ticks = np.rint(fit_time * 5).astype(int)
        # The median waveform aggregates the Train target only; actual target times are never queried at inference.
        medians = pd.DataFrame(pack['y'][:, i]).groupby(ticks, sort=True).median()
        for col in range(len(NAMES)):
            valid = medians[col].notna()
            if valid.any():
                predictions[:, i, col] = np.interp(issue_time,
                    medians.index[valid].to_numpy() / 5, medians.loc[valid, col].to_numpy())
    return predictions


def load_model_frame(data, spec, role):
    pack, meta = prepare_aligned(data, spec, role)
    if spec['family'] in ('persistence', 'template'):
        predictions = baseline_values(data, spec, role, meta)
    else:
        filename = f'{role}.npy' if spec['family'] == 'v1_archive' else f'PREDICTIONS_{role}.npy'
        path = spec['_directory'] / filename
        predictions = np.load(path, mmap_mode='r', allow_pickle=False)
    if predictions.shape != pack['y'].shape:
        raise ValueError(f'{spec["id"]}/{role} prediction shape {predictions.shape} != {pack["y"].shape}')
    return pack, meta, predictions


def valid_targets(pack, meta, predictions, horizon_idx, target_idx):
    obs = pack['y'][:, horizon_idx, target_idx]
    pred = predictions[:, horizon_idx, target_idx]
    available = pack['available_ns'][:, horizon_idx, target_idx]
    sample = pack['sample_ns'][:, horizon_idx, target_idx]
    issue = meta.issue_ns.to_numpy(np.int64)
    end = meta.end_ns.to_numpy(np.int64)
    if np.any(np.isfinite(obs) & ((available <= issue) | (available > end) | (sample > available))):
        raise ValueError('Target availability is outside the originating cycle or before observation')
    return np.isfinite(obs) & np.isfinite(pred)


def metric(err, cycles, sessions, unique=None):
    err = np.asarray(err, dtype=np.float64)
    if not len(err):
        return {'n': 0, 'cycle_n': 0, 'session_n': 0}
    abs_err = np.abs(err)
    squared = err * err
    p90, p95 = np.quantile(abs_err, [.9, .95])
    result = dict(n=len(err), mae=float(abs_err.mean()), rmse=float(np.sqrt(squared.mean())),
                  p90=float(p90), p95=float(p95), bias=float(err.mean()),
                  underprediction_fraction=float(np.mean(err < 0)))
    for key, values in [('cycle', cycles), ('session', sessions)]:
        codes, names = pd.factorize(values)
        assert (codes >= 0).all()
        counts = np.bincount(codes)
        means = np.bincount(codes, weights=abs_err) / counts
        result[f'{key}_n'] = len(names)
        result[f'{key}_equal_mae'] = float(means.mean())
        result[f'{key}_equal_rmse'] = float(np.sqrt(np.mean(np.bincount(codes, weights=squared) / counts)))
        if key == 'session':
            worst = int(np.argmax(means))
            result.update(worst_session=str(names[worst]), worst_session_mae=float(means[worst]), worst_session_n=int(counts[worst]))
    if unique is not None:
        result.update(unique_n=int(unique.sum()), unique_mae=float(abs_err[unique].mean()),
                      unique_rmse=float(np.sqrt(squared[unique].mean())))
    return result


def as_display(result, j):
    unit = NATIVE_UNITS[j]
    mult = 1000 if unit in ('A', 'V', 'g') else 1
    converted = {f'{k}_display': v * mult for k, v in result.items() if k.endswith(('mae','rmse')) or k in ('bias','p90','p95')}
    return {**result, **converted, 'unit': unit, 'display_unit': f'm{unit}' if mult == 1000 else unit}


def phase_values(seconds):
    return np.asarray(PHASES)[np.searchsorted([2.5, 5.5, 9.], seconds, side='right')]


def motion_by_target(meta, sample, values):
    samples = pd.DataFrame({'cycle_id': meta.cycle_id.to_numpy(), 'sample_ns': sample, 'value': values})
    unique = samples.drop_duplicates(['cycle_id','sample_ns']).sort_values(['cycle_id','sample_ns'])
    by_cycle = unique.groupby('cycle_id', sort=False)
    recent = by_cycle.sample_ns.diff().le(.75 * NS)
    delta = by_cycle.value.diff().abs()
    unique['motion'] = np.select([recent & delta.le(1), recent & delta.gt(1)],
        ['steady_le_1_raw','moving_gt_1_raw'], default='unknown')
    return samples.merge(unique[['cycle_id','sample_ns','motion']], on=['cycle_id','sample_ns'],
                         how='left', validate='many_to_one').motion.to_numpy()


def export_channel(meta, pack, predictions, i, j, h, tags, valid):
    sample = pack['sample_ns'][:, i, j]
    available = pack['available_ns'][:, i, j]
    issue = meta.issue_ns.to_numpy(np.int64)
    target_seq = pack['target_seq'][:, i, j]
    obs, pred = pack['y'][:, i, j], predictions[:, i, j]
    actual_valid, prediction_valid = np.isfinite(obs), np.isfinite(pred)
    deadline = issue + round((h+(tags['w'] if GROUP_NAME[j]=='A_STAT' else 0.))*NS)
    reason = np.select([valid, ~actual_valid & (deadline >= meta.end_ns.to_numpy(np.int64)), ~actual_valid],
        ['scored','nominal_target_or_window_at_or_after_end','no_matching_target_before_end'],default='forecast_missing')
    return meta[['role','cycle_id','session_id','campaign_id','group_id','label','start_pose','issue_ns','relative_s']].assign(
        **tags, target=NAMES[j], target_group=GROUP_NAME[j], unit=NATIVE_UNITS[j], horizon_s=h,
        nominal_target_ns=issue+round(h*NS), actual_target_ns=sample, target_available_ns=available,
        target_sequence_id=target_seq, target_valid_mask=actual_valid, forecast_valid=prediction_valid, scored=valid,
        score_status=reason,
        effective_horizon_s=np.where(actual_valid, (sample-issue)/NS, np.nan),
        available_horizon_s=np.where(actual_valid, (available-issue)/NS, np.nan),
        target_native_ms=pack['target_mcu_read_end_ms'][:, i, j], saturated=pack['target_saturated'][:, i, j],
        duplicate_h_archive_grid=pack['target_duplicate_h'][:, i, j], duplicate_h=pack['target_duplicate_requested_h'][:, i, j],
        duplicate_prior_issue=pack['target_duplicate_prior_issue'][:, i, j],
        actual=obs, forecast=pred)


def channel_metrics(meta, pack, predictions, i, j, h, tags, denom, strata, sessions, full_strata):
    valid = valid_targets(pack, meta, predictions, i, j)
    obs, pred = pack['y'][:, i, j], predictions[:, i, j]
    sample = pack['sample_ns'][:, i, j]
    available = pack['available_ns'][:, i, j]
    seq = pack['target_seq'][:, i, j]
    end, issue, begin = [meta[c].to_numpy(np.int64) for c in ('end_ns','issue_ns','No3_begin_ns')]
    target_ns = issue + round(h * NS)
    labels = meta.label.to_numpy()
    tags = dict(tags, target=NAMES[j], target_group=GROUP_NAME[j], horizon_s=h)
    output, coverage = [], []
    for label in pd.unique(labels):
        group = labels == label
        use = group & valid
        rows = meta.loc[use].reset_index(drop=True)
        # Earliest-issued scored prediction represents each unique physical observation.
        unique = ~pd.DataFrame({'cycle':rows.cycle_id,'sample':sample[use],'seq':seq[use]}).duplicated().to_numpy()
        err = pred[use] - obs[use]
        out = as_display(metric(err, rows.cycle_id, rows.session_id, unique), j)
        output.append(dict(tags, label=label, scope='native', **out))
        if len(err):
            if GROUP_NAME[j] in ('A_RAW','A_STAT'):
                nonsat = ~pack['target_saturated'][use, i, j]
                if not nonsat.all():
                    # Compute unique membership after removing saturated target samples.
                    uu = ~pd.DataFrame({'cycle':rows.cycle_id[nonsat], 'sample':sample[use][nonsat],
                        'seq':seq[use][nonsat]}).duplicated().to_numpy()
                    output.append(dict(tags, label=label, scope='without_recorded_saturation',
                        **as_display(metric(err[nonsat], rows.cycle_id[nonsat], rows.session_id[nonsat], uu), j)))
        if len(err) and (label == 'normal' or full_strata):
            dimensions = {'phase':phase_values((sample[use]-begin[use])/NS),
                          'issue_phase':phase_values((issue[use]-begin[use])/NS),
                          'start_pose':rows.start_pose.fillna('unknown').to_numpy(),
                          'power_context':rows.power_context.fillna('unknown').to_numpy(),
                          'block_start':pd.cut(rows.block_local_cycle, [0,5,10,np.inf], labels=['1–5','6–10','11+']).astype(str).to_numpy()}
            if GROUP_NAME[j] == 'P':
                dimensions['motion'] = motion_by_target(rows, sample[use], obs[use])
            if GROUP_NAME[j] == 'A_STAT':
                dimensions['amplitude_vs_normal_dev'] = np.where(obs[use] > tags['stat_dev_p95'], 'high', 'ordinary')
            for name, values in dimensions.items():
                for value in pd.unique(values):
                    mask = values == value
                    strata.append(dict(tags, label=label, dimension=name, value=value,
                        **as_display(metric(err[mask], rows.cycle_id[mask], rows.session_id[mask]), j)))
            for session in pd.unique(rows.session_id):
                mask = rows.session_id.eq(session).to_numpy()
                sessions.append(dict(tags, label=label, session_id=session,
                    campaign_id=rows.loc[mask,'campaign_id'].iloc[0],
                    **as_display(metric(err[mask], rows.cycle_id[mask], rows.session_id[mask]), j)))
        lag = (sample[use]-target_ns[use])/NS
        observed_h = (sample[use]-issue[use])/NS
        receipt_h = (available[use]-issue[use])/NS
        diag = pd.DataFrame({'cycle':rows.cycle_id, 'issue':rows.relative_s, 'available':(available[use]-begin[use])/NS})
        first = diag.groupby('cycle',sort=False).min()
        median = lambda values: float(np.median(values)) if len(values) else np.nan
        quant = lambda values, q: float(np.quantile(values,q)) if len(values) else np.nan
        stat_width = tags['w'] if GROUP_NAME[j] == 'A_STAT' else 0.
        ended = group & (target_ns + round(stat_width*NS) >= end)
        coverage.append(dict(tags, label=label, full_issue_grid_n=denom.get(label, 0), history_valid_issue_n=int(group.sum()),
            target_n=int((group & np.isfinite(obs)).sum()), prediction_n=int((group & np.isfinite(pred)).sum()),
            scored_n=int(use.sum()), history_valid_but_past_end_n=int(ended.sum()),
            history_valid_but_target_missing_inside_cycle_n=int((group & ~ended & ~np.isfinite(obs)).sum()),
            coverage_all_issue_grid=float(use.sum()/denom[label]) if denom.get(label) else np.nan,
            coverage_history_valid=float(use.sum()/group.sum()), cycle_n=rows.cycle_id.nunique(),
            duplicate_across_h_n=int(pack['target_duplicate_requested_h'][use,i,j].sum()),
            duplicate_across_archive_h_n=int(pack['target_duplicate_h'][use,i,j].sum()),
            duplicate_prior_issue_n=int(pack['target_duplicate_prior_issue'][use,i,j].sum()),
            unique_observation_n=int(unique.sum()), first_issue_median_s=first.issue.median(), first_available_median_s=first.available.median(),
            effective_horizon_p05_s=quant(observed_h,.05), effective_horizon_median_s=median(observed_h),
            effective_horizon_p95_s=quant(observed_h,.95), nominal_error_median_s=median(lag),
            nominal_error_p95_s=quant(lag,.95), available_horizon_median_s=median(receipt_h),
            available_horizon_p95_s=quant(receipt_h,.95)))
    return output, coverage, valid


def layout_refs(data, specs):
    cache = {}
    for spec in specs:
        layout = key_for(spec)
        if layout not in cache:
            pack, _ = prepare_aligned(data, spec, 'train')
            dev, _ = prepare_aligned(data, spec, 'development')
            cache[layout] = (np.maximum(np.nanstd(pack['y'], axis=0), TARGET_FLOOR),
                             np.nanquantile(dev['y'], .95, axis=0))
    return cache


def paired_series(pack, meta, pred, i, j, chosen):
    valid = valid_targets(pack, meta, pred, i, j) & chosen
    p, y = pred[valid, i, j], pack['y'][valid, i, j]
    sample, seq = [pack[key][valid, i, j] for key in ('sample_ns','target_seq')]
    table = meta.loc[valid, ['cycle_id','issue_ns','session_id','No3_begin_ns']].copy()
    table['sample_ns'], table['target_seq'], table['y'], table['pred'] = sample, seq, y, p
    table['target_elapsed_s'] = (sample - table.No3_begin_ns.to_numpy(np.int64)) / NS
    return table


def group_native_comparison(selected, data, specs, root):
    """Emit exactly matched, row-level visual examples at naturally occurring poses."""
    by_id = {s['id']:s for s in specs}
    representatives = pd.read_csv(root/'SPLIT_MANIFEST.csv',low_memory=False).query('role == "evaluation"')
    representatives = representatives.sort_values(['label','start_pose','cycle_id']).drop_duplicates(['label','start_pose'])
    keep = set(representatives.cycle_id)
    records = []
    for name in list(selected)+['v1_mt_f2_tree']:
        if name not in by_id:
            continue
        spec = by_id[name]
        _,horizons,_ = key_for(spec)
        if .5 not in horizons:
            continue
        pack,meta,pred = load_model_frame(data,spec,'evaluation')
        i = horizons.index(.5)
        chosen = meta.cycle_id.isin(keep)
        for j in range(len(NAMES)):
            valid = valid_targets(pack,meta,pred,i,j)
            table = export_channel(meta,pack,pred,i,j,.5,model_tags(spec),valid).loc[chosen].copy()
            table['nominal_target_elapsed_s'] = table.relative_s+.5
            table['actual_target_elapsed_s'] = table.relative_s + table.effective_horizon_s
            table['available_elapsed_s'] = table.relative_s + table.available_horizon_s
            records.append(table)
    if records:
        write_csv(pd.concat(records,ignore_index=True),'REPRESENTATIVE_WAVEFORMS',root)


def paired_result(own, ref, tags, policy):
    keys = ['cycle_id','issue_ns']
    if policy == 'same_issue_and_target_id':
        keys += ['sample_ns','target_seq']
    if policy == 'common_target_second_bins':
        # Both horizons may occur in the same elapsed-time bin with different physical target samples.
        # Restrict each side to joint cycle/bin support without representing the observations as paired.
        own = own.assign(elapsed_bin=np.floor(own.target_elapsed_s).astype(int))
        ref = ref.assign(elapsed_bin=np.floor(ref.target_elapsed_s).astype(int))
        pair_keys = ['cycle_id','elapsed_bin']
        shared = own[pair_keys].drop_duplicates().merge(ref[pair_keys].drop_duplicates(),on=pair_keys)
        own, ref = [part.merge(shared,on=pair_keys,validate='many_to_one') for part in (own,ref)]
        e, er = (part.pred.to_numpy()-part.y.to_numpy() for part in (own,ref))
        n, n_ref = len(e), len(er)
        exact = np.nan
    elif policy == 'same_issue_and_target_id' and len(own) == len(ref) and all(
            np.array_equal(own[col].to_numpy(),ref[col].to_numpy()) for col in keys):
        np.testing.assert_allclose(own.y.to_numpy(),ref.y.to_numpy(),rtol=0,atol=1e-8)
        e, er = (part.pred.to_numpy()-part.y.to_numpy() for part in (own,ref))
        n = n_ref = exact = len(e)
    else:
        both = own.merge(ref, on=keys, suffixes=('', '_reference'), validate='one_to_one', sort=False)
        if policy == 'same_issue_and_target_id':
            np.testing.assert_allclose(both.y.to_numpy(), both.y_reference.to_numpy(), rtol=0, atol=1e-8)
        e = both.pred.to_numpy()-both.y.to_numpy()
        er = both.pred_reference.to_numpy()-both.y_reference.to_numpy()
        n = n_ref = len(e)
        exact = n if policy == 'same_issue_and_target_id' else int((both.sample_ns == both.sample_ns_reference).sum())
    if not n or not n_ref:
        return None
    j = NAMES.index(tags['target'])
    mult = 1000 if NATIVE_UNITS[j] in ('A','V','g') else 1
    mae, ref_mae = [float(np.mean(np.abs(e_))) for e_ in (e,er)]
    return dict(tags, scope=policy, matched_n=n, reference_n=n_ref, exact_target_matches=exact,
                mae=mae, reference_mae=ref_mae, delta_mae=mae-ref_mae, mae_display=mae*mult,
                reference_mae_display=ref_mae*mult, display_unit=f'm{NATIVE_UNITS[j]}' if mult == 1000 else NATIVE_UNITS[j])


def choose_pair_specs(spec, specs):
    L, hs, w = key_for(spec)
    refs = []
    for ref in specs:
        if ref['id'] == spec['id']:
            continue
        rL, rhs, rw = key_for(ref)
        same_layout = (L,hs,w)==(rL,rhs,rw)
        same_input = spec.get('group') == ref.get('group')
        both_o3 = spec.get('output') == ref.get('output') == 'O3'
        compare = (ref['family'] in ('persistence','template') and same_layout or
                   ref['id'] in ('v1_e_f4_tree','v1_mt_f2_tree','v1_mt_f2_ridge','v1_mt_f2_gru16') and .5 in hs or
                   same_layout and same_input and ref['family'] in ('ridge','hgb') and ref.get('output')=='O3'
                   and spec['family'] not in ('ridge','hgb','persistence','template','v1_archive'))
        shared_arch = all(spec.get(k)==ref.get(k) for k in ('family','size','seed'))
        if shared_arch and spec['family'] not in ('persistence','template','v1_archive'):
            compare |= same_input and both_o3 and ((rL == 5 and L != 5) or w != rw)
            compare |= L==rL==5 and both_o3 and not same_input and same_layout and ref.get('group')=='F0'
            compare |= same_input and both_o3 and L==rL and w==rw and len(hs)>1 and rhs==(.5,)
            compare |= same_input and spec.get('output') != 'O3' and ref.get('output') == 'O3' and same_layout
            compare |= ref.get('group')=='F4' and not same_input and same_layout and both_o3
        compare |= (all(spec.get(k)==ref.get(k) for k in ('family','group','seed','output'))
                    and spec.get('size')=='large' and ref.get('size')=='base' and same_layout)
        if compare:
            refs.append(ref)
    return refs


def compared(data, spec, refs, role, current):
    pack, meta, pred = current
    h_list = key_for(spec)[1]
    selected = meta.label.eq('normal').to_numpy()
    tags = model_tags(spec)
    matched = []
    own_series = {(i,j):paired_series(pack,meta,pred,i,j,selected) for i,_ in enumerate(h_list) for j,_ in enumerate(NAMES)}
    for ref in refs:
        reference = load_model_frame(data, ref, role)
        rp, rm, rv = reference
        r_selected = rm.label.eq('normal').to_numpy()
        ref_hs = key_for(ref)[1]
        for i, h in enumerate(h_list):
            if h not in ref_hs:
                continue
            r_i = ref_hs.index(h)
            for j, name in enumerate(NAMES):
                own = own_series[i,j]
                if own.empty:
                    continue
                other = paired_series(rp, rm, rv, r_i, j, r_selected)
                m = paired_result(own,other,dict(tags,role=role, reference=ref['id'], target=name,
                    horizon_s=h, reference_horizon_s=h), 'same_issue_and_target_id')
                if m:
                    matched.append(m)
    if len(h_list) > 1:
        for i, h in enumerate(h_list[1:],1):
            for j, name in enumerate(NAMES):
                own = own_series[i,j]
                ref = own_series[0,j]
                for policy in ('common_issue_only','common_target_second_bins'):
                    m = paired_result(own,ref,dict(tags,role=role,reference=spec['id'],target=name,
                        horizon_s=h,reference_horizon_s=h_list[0]),policy)
                    if m:
                        matched.append(m)
    return matched


def target_group_match_index(table):
    # Keep full per-target matching in the source table and summarize only like-for-like MAEs here.
    keys = ['model_id','reference','role','horizon_s','reference_horizon_s','scope','target_group']
    grouped = table.groupby(keys,sort=False,dropna=False).agg(
        targets=('target','size'), minimum_target_n=('matched_n','min'),
        normalized_mae=('normalized_mae','mean'), reference_normalized_mae=('reference_normalized_mae','mean'),
        normalized_delta=('normalized_delta','mean')).reset_index()
    overall = grouped.groupby(keys[:-1],sort=False,dropna=False).agg(
        groups=('target_group','size'), minimum_target_n=('minimum_target_n','min'),
        normalized_mae=('normalized_mae','mean'), reference_normalized_mae=('reference_normalized_mae','mean'),
        normalized_delta=('normalized_delta','mean')).reset_index()
    return pd.concat([grouped,overall.assign(target_group='sensor_equal')],ignore_index=True)


def uncertainty(root, specs, selected, data):
    """Resample sessions, using paired cycle-equal absolute-error deltas within each."""
    specs = {s['id']:s for s in specs}
    reference = 'v1_mt_f2_tree'
    if reference not in specs:
        return
    data_ref = load_model_frame(data,specs[reference],'evaluation')
    rp,rm,rv = data_ref
    r_normal = rm.label.eq('normal').to_numpy()
    rng = np.random.default_rng(7181)
    records = []
    for name in selected:
        spec = specs[name]
        _,hs,w = key_for(spec)
        if .5 not in hs or w!=.2:
            continue
        pack,meta,pred = load_model_frame(data,spec,'evaluation')
        normal = meta.label.eq('normal').to_numpy()
        i = hs.index(.5)
        for j,target in enumerate(NAMES):
            left = paired_series(pack,meta,pred,i,j,normal)
            right = paired_series(rp,rm,rv,0,j,r_normal)
            both = left.merge(right,on=['cycle_id','issue_ns','sample_ns','target_seq'],suffixes=('','_reference'),validate='one_to_one')
            if both.empty:
                continue
            assert both.session_id.eq(both.session_id_reference).all()
            delta = np.abs(both.pred-both.y)-np.abs(both.pred_reference-both.y_reference)
            sessions = both.assign(delta=delta).groupby(['session_id','cycle_id']).delta.mean().groupby(level=0).mean().to_numpy()
            sims = rng.choice(sessions,size=(2000,len(sessions)),replace=True).mean(axis=1)
            lo,hi = np.quantile(sims,[.025,.975])
            records.append({'model_id':name,'reference':reference,'target':target,'unit':NATIVE_UNITS[j],
                            'matched_issue_n':len(both),'session_n':len(sessions),'cycle_equal_then_session_equal_delta_mae':sessions.mean(),
                            'session_bootstrap_p025':lo,'session_bootstrap_p975':hi,'resamples':2000})
    write_csv(records,'SELECTED_SESSION_BOOTSTRAP',root)


def summarize(rows):
    by = ['model_id','family','size','input','output','L','w','horizons','seed','role','horizon_s','target_group','label']
    return rows.groupby(by,dropna=False,sort=False).agg(
        group_normalized_mae=('normalized_mae','mean'), target_n=('target','size'),
        mean_coverage=('coverage_all_issue_grid','mean')).reset_index()


def vector_metric(pack,meta,pred,i,h,tags):
    cols = GROUPS['A_RAW']
    y,p = pack['y'][:,i,cols],pred[:,i,cols]
    valid = (np.isfinite(y) & np.isfinite(p)).all(axis=1)
    if not valid.any():
        return []
    measured = np.linalg.norm((p-y).astype(float),axis=1)
    sat = pack['target_saturated'][:,i,cols].any(axis=1)
    records = []
    for label in pd.unique(meta.label):
        for scope, endpoint in [('native_vector',np.ones(len(meta),dtype=bool)),('without_saturation_vector',~sat)]:
            selected = valid & meta.label.eq(label).to_numpy() & endpoint
            if not selected.any() or scope=='without_saturation_vector' and not (valid & meta.label.eq(label).to_numpy() & sat).any():
                continue
            rows = meta.loc[selected]
            records.append(dict(tags,label=label,horizon_s=h,target='XYZ_vector_error',target_group='A_RAW',scope=scope,
                                **as_display(metric(measured[selected],rows.cycle_id,rows.session_id),cols[0])))
    return records


def assemble(root, target_rows, coverage, strata, sessions, matches):
    target = pd.DataFrame(target_rows)
    coverage = pd.DataFrame(coverage)
    merged = target.loc[target.scope.eq('native') & target.n.gt(0)].merge(coverage[['model_id','role','target','horizon_s','label','coverage_all_issue_grid']],
        on=['model_id','role','target','horizon_s','label'], validate='one_to_one')
    group = summarize(merged)
    normal = group.loc[group.label.eq('normal')].copy()
    normal['sensor_equal_score'] = normal.groupby(['model_id','role','horizon_s'],sort=False).group_normalized_mae.transform('mean')
    is_anchor = normal.L.eq(2.) & normal.horizon_s.eq(.5) & normal.w.eq(.2)
    base = normal.loc[is_anchor & normal.output.eq('O3')]
    seed_keys = ['family','size','input','output','L','w','horizons','role','horizon_s','target_group']
    seed_groups = normal.loc[normal.seed.isin([7181,7182,7183])].groupby(seed_keys,dropna=False,sort=False)
    seed_variation = seed_groups.agg(seed_count=('seed','nunique'), normalized_mae_min=('group_normalized_mae','min'),
        normalized_mae_median=('group_normalized_mae','median'),normalized_mae_max=('group_normalized_mae','max'),
        normalized_mae_std=('group_normalized_mae','std')).reset_index()
    repeated = seed_variation.loc[seed_variation.seed_count.gt(1)]
    seed_rows = normal.merge(repeated,on=seed_keys,how='inner',validate='many_to_one')
    stages = {'MODEL_CAPACITY_COMPARISON':base.loc[base.input.eq('F4') | base.family.isin(['persistence','template','v1_archive'])],
              'INPUT_SENSOR_ABLATION':base, 'WINDOW_HORIZON_COMPARISON':normal,
              'OUTPUT_GROUP_COMPARISON':normal.loc[is_anchor], 'SEED_STABILITY':seed_rows}
    for name, rows in stages.items():
        write_csv(rows, name, root)
    paired = pd.DataFrame(matches)
    scales = target.loc[target.scope.eq('native'),['model_id','role','horizon_s','target','train_scale','target_group']].drop_duplicates()
    paired = paired.merge(scales,on=['model_id','role','horizon_s','target'],validate='many_to_one')
    paired['normalized_mae'] = paired.mae/paired.train_scale
    paired['reference_normalized_mae'] = paired.reference_mae/paired.train_scale
    paired['normalized_delta'] = paired.normalized_mae-paired.reference_normalized_mae
    for name, rows in [('METRICS_BY_TARGET',target),('COVERAGE_AND_TARGET_LAG',coverage),
                       ('METRICS_BY_PHASE_AND_POSE',strata),('METRICS_BY_SESSION',sessions),('MATCHED_TARGET_COMPARISON',matches)]:
        write_csv(paired if name=='MATCHED_TARGET_COMPARISON' else rows, name, root)
    write_csv(target_group_match_index(paired),'MATCHED_SENSOR_GROUP_COMPARISON',root)
    return target, coverage


def run(root=ROOT, selected=(), development_only=False, evaluation_only=False):
    root = Path(root)
    (root/'metrics').mkdir(exist_ok=True, parents=True)
    data = Data(root)
    specs = run_specs(root)
    grid = root/'grids'/'ALL.json'
    if not development_only and grid.is_file():
        expected = {item['id'] for item in read_json(grid)}
        trained = {spec['id'] for spec in specs}
        if trained != expected:
            raise ValueError(f'Sealed grid and completed model sets differ: missing={expected-trained}, extra={trained-expected}')
    specs += archived_specs(root)
    specs += basic_specs(specs)
    names = {s['id'] for s in specs}
    assert set(selected).issubset(names)
    detailed = set(selected) | {s['id'] for s in specs if s['family'] == 'v1_archive'}
    roles = ['development'] if development_only else ['evaluation'] if evaluation_only else ['development','evaluation']
    layout = layout_refs(data,specs)
    target_rows, coverage, strata, sessions, matches = [], [], [], [], []
    raw_path = root/'metrics'/'FORECASTS_AND_TARGETS.csv.gz'
    raw_tmp = raw_path.with_suffix('.gz.tmp')
    raw = gzip.open(raw_tmp,'wt',compresslevel=1,encoding='utf-8-sig',newline='') if not development_only else None
    first_export = True
    replay_rows = 0
    try:
        for role in roles:
            base_meta, _ = data.direct(role)
            denom = base_meta.groupby('label').size().to_dict()
            for spec in specs:
                if spec.get('_directory') and spec['family'] != 'v1_archive' and not (spec['_directory']/f'PREDICTIONS_{role}.npy').exists():
                    raise FileNotFoundError(f'Missing {role} predictions for {spec["id"]}')
                current = load_model_frame(data,spec,role)
                pack, meta, pred = current
                verify_forecast_predictions(spec,meta,pack,pred)
                if role == 'development' and not meta.label.eq('normal').all():
                    raise ValueError('Development contains abnormal cycles')
                scales, dev_p95 = layout[key_for(spec)]
                tags = dict(model_tags(spec),role=role)
                for i,h in enumerate(key_for(spec)[1]):
                    target_rows.extend(vector_metric(pack,meta,pred,i,h,tags))
                    for j,name in enumerate(NAMES):
                        if not np.isfinite(pred[:,i,j]).any():
                            continue
                        ch_tags = dict(tags,train_scale=scales[i,j],stat_dev_p95=dev_p95[i,j])
                        metrics, diagnostics, valid = channel_metrics(meta,pack,pred,i,j,h,ch_tags,denom,strata,sessions,spec['id'] in detailed)
                        target_rows.extend(dict(m, normalized_mae=m.get('mae',np.nan)/scales[i,j]) for m in metrics)
                        coverage.extend(diagnostics)
                        if raw is not None and spec['id'] in detailed and role=='evaluation':
                            channel = export_channel(meta,pack,pred,i,j,h,tags,valid)
                            channel.to_csv(raw,header=first_export,index=False,float_format='%.9g')
                            first_export = False
                            replay_rows += len(channel)
                matches.extend(compared(data,spec,choose_pair_specs(spec,specs),role,current))
                print(f'{role} {spec["id"]}: {len(meta):,} issues',flush=True)
    finally:
        if raw:
            raw.close()
    if raw:
        raw_tmp.replace(raw_path)
    target, coverage = assemble(root,target_rows,coverage,strata,sessions,matches)
    if not development_only:
        uncertainty(root,specs,selected,data)
        group_native_comparison(selected,data,specs,root)
        report = {'prediction_model_count':len(specs),'target_metrics_rows':len(target),'coverage_rows':len(coverage),
                  'exact_and_auxiliary_comparisons':len(matches),'long_export_rows':replay_rows,
                  'long_export_model_ids':sorted(detailed),'long_export_role':'evaluation',
                  'all_other_exact_predictions':'runs/<id>/PREDICTIONS_<role>.npy',
                  'stratified_anomaly_model_ids':sorted(detailed),
                  'integer_source_target_and_availability_clock':True,'targets_keyed_by_cycle_issue_horizon_output':True,
                  'eligible_target_counts_are_output_specific':True,'unique_target_rule':'earliest valid issue per cycle, target host sample time and target sequence within each horizon',
                  'position_motion_strata':'evaluation-only change >1 raw across consecutive unique position responses separated by no more than .75 s',
                  'coverage_denominator':'all active No.3 0.2 s issue slots, including early and late positions with no valid forecast',
                  'phase_basis':'phase uses actual target/sample elapsed; issue_phase uses forecast issue elapsed',
                  'train_only_normalization_and_template':True,'evaluation_role_excluded_from_selection':True}
        (root/'metrics'/'PREDICTION_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev',action='store_true')
    parser.add_argument('--evaluation-only',action='store_true')
    parser.add_argument('--selected',nargs='*',default=[])
    parser.add_argument('--root',type=Path,default=ROOT)
    args = parser.parse_args()
    if args.dev and args.evaluation_only:
        parser.error('--dev and --evaluation-only are mutually exclusive')
    run(args.root,args.selected,args.dev,args.evaluation_only)


if __name__ == '__main__':
    main()
