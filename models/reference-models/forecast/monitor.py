"""Causal residual monitoring and coverage-matched offline replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import Data, V1_ELECTRIC, V1_MULTI
from evaluate import (ROOT, NS, STRIDE_NS, GROUPS, NAMES, archived_specs,
                      basic_specs, key_for, load_model_frame, run_specs, write_csv)


DIRECT = 'direct_recent_rms_peak'
FROZEN_LOGISTIC = 'frozen_logistic_cycle_end'
MONITOR_H = .5
TARGET_FLOOR = np.array([1e-6, 1e-6, *([1e-3] * 6), *([1e-6] * 6)])
POLICIES = {**GROUPS, 'ALL': list(range(len(NAMES)))}
WINDOW_NS = STRIDE_NS


def event_components(data, spec, role):
    if spec['id'] == DIRECT:
        meta, values = data.direct(role)
        clock = meta.issue_ns.to_numpy(np.int64)
        arrived = np.broadcast_to(clock[:, None], values.shape)
        return meta, values, arrived, arrived, np.full(values.shape,-1), values, np.full(values.shape,np.nan)
    pack, meta, prediction = load_model_frame(data,spec,role)
    i = key_for(spec)[1].index(MONITOR_H)
    actual, prediction = pack['y'][:,i], prediction[:,i]
    available, sample, seq = (pack[key][:,i] for key in ('available_ns','sample_ns','target_seq'))
    issue = meta.issue_ns.to_numpy(np.int64)
    end = meta.end_ns.to_numpy(np.int64)
    minimum = issue[:,None] + round(MONITOR_H*NS)
    endpoint = np.broadcast_to(minimum, available.shape).copy()
    endpoint[:,GROUPS['A_STAT']] += round(key_for(spec)[2]*NS)
    valid = np.isfinite(actual) & np.isfinite(prediction)
    if np.any(valid & ((available < endpoint) | (available > end[:,None]) | (sample > available))):
        raise ValueError('Residual would use a target before availability or after No.3')
    residual = np.where(valid,np.abs(actual.astype(float)-prediction.astype(float)),np.nan)
    return meta,residual,available,sample,seq,actual,prediction


def fit_scales(parts, direct):
    meta,values,*_ = parts
    if not meta.role.eq('development').all() or not meta.label.eq('normal').all():
        raise ValueError('Only normal Development may scale monitoring scores')
    floor = np.array([1e-6,1e-6]) if direct else TARGET_FLOOR
    return np.array([max(float(np.quantile(x[np.isfinite(x)],.90,method='higher')), floor[j])
                     if np.isfinite(x).any() else np.nan for j,x in enumerate(values.T)])


def policy_events(parts, cols, scales, direct=False):
    meta,values,available,sample,seq,actual,pred = parts
    norm = values[:,cols] / scales[cols]
    row, subcol = np.where(np.isfinite(norm))
    col = np.asarray(cols)[subcol]
    selected = meta.iloc[row].reset_index(drop=True)
    names = ['current_RMS_g','current_Peak_g'] if direct else NAMES
    issue = selected.issue_ns.to_numpy(np.int64)
    arrival = available[row,col]
    origin = selected.No3_begin_ns.to_numpy(np.int64)
    nominal = issue if direct else issue + round(MONITOR_H*NS)
    events = pd.DataFrame({'cycle_id':selected.cycle_id,'issue_ns':issue,'available_ns':arrival,
        'actual_target_ns':sample[row,col], 'target_sequence_id':seq[row,col],
        'nominal_target_ns':nominal,
        'clock_slot':(arrival-origin)//WINDOW_NS, 'score':norm[row,subcol],
        'target':np.asarray(names)[col], 'absolute_residual_or_direct_value':values[row,col],
        'actual':actual[row,col], 'forecast':pred[row,col]})
    # All newly arrived residuals compete at the same receive time. No older score is carried forward.
    maxima = events.groupby(['cycle_id','available_ns'],sort=False).score.idxmax()
    return events.loc[maxima].sort_values(['cycle_id','available_ns'],kind='stable').reset_index(drop=True)


def threshold(calibration, expected, scope):
    maxima = calibration.groupby('cycle_id',sort=False).score.max()
    if not set(maxima.index).issubset(expected) or maxima.empty:
        raise ValueError(f'Invalid calibration scope {scope}')
    value = float(np.quantile(maxima,.95,method='higher'))
    return value, {'threshold':value,'calibration_cycles_valid':len(maxima),
                   'calibration_cycles_total':len(expected),'calibration_fp':int(maxima.gt(value).sum())}


def common_time(own, other):
    keys = ['cycle_id','clock_slot']
    common = own[keys].drop_duplicates().merge(other[keys].drop_duplicates(), on=keys,validate='one_to_one')
    return [frame.merge(common,on=keys,validate='many_to_one') for frame in (own,other)]


def to_cycles(cycles, arrivals, limit, tag):
    indices = ['cycle_id','session_id','campaign_id','group_id','label','start_pose','power_context','No3_begin_ns','No3_end_ns']
    elapsed = arrivals.merge(cycles[['cycle_id','No3_begin_ns','No3_end_ns']],on='cycle_id',validate='many_to_one')
    if np.any(elapsed.available_ns.gt(elapsed.No3_end_ns)):
        raise ValueError('A scoring event is after end of No.3')
    elapsed['arrival_s'] = (elapsed.available_ns-elapsed.No3_begin_ns)/NS
    elapsed['alert'] = elapsed.score.gt(limit) & elapsed.available_ns.lt(elapsed.No3_end_ns)
    valid = elapsed.groupby('cycle_id',sort=False).agg(event_n=('score','size'),score_max=('score','max'),
        first_score_s=('arrival_s','min'),last_score_s=('arrival_s','max'),valid_clock_slots=('clock_slot','nunique'))
    unique_bins = elapsed.drop_duplicates(['cycle_id','clock_slot'])
    lo = unique_bins.clock_slot * WINDOW_NS
    hi = np.minimum(lo+WINDOW_NS, unique_bins.No3_end_ns-unique_bins.No3_begin_ns)
    unique_bins = unique_bins.assign(slot_seconds=np.maximum(hi-lo,0)/NS)
    valid = valid.join(unique_bins.groupby('cycle_id').slot_seconds.sum().rename('covered_s'))
    alert = elapsed.loc[elapsed.alert].sort_values(['available_ns','issue_ns']).drop_duplicates('cycle_id').set_index('cycle_id')
    first_columns = ['available_ns','issue_ns','nominal_target_ns','actual_target_ns','target_sequence_id']
    first = alert[first_columns].astype('Int64')
    first = first.join(alert[['arrival_s','score','target','actual','forecast','absolute_residual_or_direct_value']])
    first = first.add_prefix('first_alert_')
    base = cycles[indices].join(valid,on='cycle_id').join(first,on='cycle_id')
    base['alert'] = base.first_alert_available_ns.notna()
    base['monitor_available'] = base.event_n.notna()
    for col in ('event_n','valid_clock_slots','covered_s'):
        base[col] = base[col].fillna(0)
    base['duration_s'] = (base.No3_end_ns-base.No3_begin_ns)/NS
    base['seconds_before_end'] = (base.No3_end_ns-base.first_alert_available_ns)/NS
    base['head_unmonitored_s'] = base.first_score_s.fillna(base.duration_s)
    base['event_time_status'] = np.select([base.label.eq('normal'),base.label.isin(['A121','B82','C182'])],
        ['not_applicable','condition_present_before_start'],default='physical_event_time_unavailable')
    # Nominal protocol regions are only anchors, not physical-event timing observations.
    early_contact = base.label.eq('foam_contact') & base.first_alert_arrival_s.lt(4.5)
    early_disturbance = base.label.eq('station_disturbance') & base.first_alert_arrival_s.lt(10.0)
    base['before_observed_region'] = np.select([early_contact,early_disturbance],
        ['before_nominal_contact_region','before_nominal_disturbance_region'],default='not_flagged')
    return base.assign(**tag, threshold=limit)


def summarize(rows, keys):
    records = []
    for indexes, chunk in rows.groupby(keys,dropna=False,sort=False):
        marked = chunk.loc[chunk.alert]
        monitored = int(chunk.monitor_available.sum())
        counts = {**dict(zip(keys,indexes if isinstance(indexes,tuple) else (indexes,))),
                  'cycle_n':len(chunk),'monitor_available_n':monitored,'unavailable_n':len(chunk)-monitored,
                  'alert_n':len(marked),'alert_rate_all':len(marked)/len(chunk),
                  'alert_rate_monitored':len(marked)/monitored if monitored else np.nan,
                  'covered_s':chunk.covered_s.sum(),'coverage_time':chunk.covered_s.sum()/chunk.duration_s.sum(),
                  'median_first_score_s':chunk.first_score_s.median(),'median_head_unmonitored_s':chunk.head_unmonitored_s.median(),
                  'median_first_alert_s':marked.first_alert_arrival_s.median(),'median_seconds_before_end':marked.seconds_before_end.median(),
                  'earlier_than_nominal_region_n':int(marked.before_observed_region.ne('not_flagged').sum())}
        records.append(counts)
    return pd.DataFrame(records)


def timing_counts(rows):
    early = rows.loc[rows.alert & rows.before_observed_region.ne('not_flagged')]
    return early.groupby(['model_id','policy','label','before_observed_region','first_alert_target'],sort=False).size().rename('cycles').reset_index()


def metric_tables(root, cycle_data, matched_rows, thresholds):
    rows = pd.concat(cycle_data,ignore_index=True)
    dimensions = ['model_id','policy','scope','label']
    native = rows.loc[rows.scope.eq('native')].copy()
    native_metrics = summarize(native,dimensions)
    by_pose = summarize(native,dimensions+['start_pose'])
    summary = pd.concat([native_metrics,by_pose],ignore_index=True)
    write_csv(summary,'RESIDUAL_ALERT_METRICS',root)
    write_csv(rows,'FIRST_ALERT_TIMES',root)
    write_csv(pd.concat(matched_rows,ignore_index=True) if matched_rows else [],'MATCHED_COVERAGE_ALERT_COMPARISON',root)
    write_csv(thresholds,'MONITOR_CALIBRATION',root)
    delta_rows = []
    columns = ['cycle_id','alert','monitor_available']
    for (model,label), block in native.groupby(['model_id','label'],sort=False):
        base = block.loc[block.policy.eq('E'),columns]
        if base.empty:
            continue
        for policy, chunk in block.groupby('policy',sort=False):
            if policy == 'E':
                continue
            joined = chunk[columns].merge(base,on='cycle_id',suffixes=('','_E'),validate='one_to_one')
            matched = joined.loc[joined.monitor_available & joined.monitor_available_E]
            delta_rows.append({'model_id':model,'label':label,'policy':policy,'matched_cycle_n':len(matched),
                'added_alert_n':int((matched.alert & ~matched.alert_E).sum()),'E_only_alert_n':int((~matched.alert & matched.alert_E).sum())})
    write_csv(delta_rows,'SENSOR_GROUP_ALERT_CHANGES_VS_E',root)
    first = native.loc[native.alert].groupby(dimensions+['start_pose','first_alert_target'],dropna=False,sort=False).agg(
        alerts=('cycle_id','size'), median_first_s=('first_alert_arrival_s','median'),
        median_actual=('first_alert_actual','median'),median_forecast=('first_alert_forecast','median')).reset_index()
    write_csv(first,'FIRST_ALERT_TARGETS_BY_POSE',root)
    write_csv(timing_counts(native),'EARLY_RELATIVE_TO_PROTOCOL_REGION',root)
    return native


def reference_logistic(rows, root):
    source = V1_ELECTRIC/'metrics'/'FIRST_ALERT_TIMES.csv'
    with source.open('rb') as stream:
        original_sha = hashlib.file_digest(stream,'sha256').hexdigest()
    bundle = V1_ELECTRIC.parent.parent/'No3_bolt_normal_retrain_20261111'/'analysis'/'retrained_bolt_normal'/'models'/'LOGISTIC_BOLT_NORMAL_EXPANDED.joblib'
    with bundle.open('rb') as stream:
        bundle_sha = hashlib.file_digest(stream,'sha256').hexdigest()
    if bundle_sha != 'd13f356a01973a4fbb188396301b9c98f9062a36d24d35b7421031907bde548d':
        raise ValueError('Frozen logistic weight hash differs from the V1 audit')
    old = pd.read_csv(source,usecols=['cycle_id','old_score','old_threshold','old_alarm','frozen_cycle_role']).drop_duplicates()
    if not old.cycle_id.is_unique or old.old_threshold.nunique() != 1:
        raise ValueError('Frozen logistic archive contains conflicting scores')
    if not np.all(old.old_alarm.eq(old.old_score.gt(old.old_threshold))) or not np.isclose(old.old_threshold.iloc[0],.8214447226813868):
        raise ValueError('Frozen logistic archive alarms do not match the stored threshold')
    write_csv(old,'FROZEN_LOGISTIC_CYCLE_REFERENCE',root)
    paired = rows.merge(old,on='cycle_id',how='left',validate='many_to_one')
    records = []
    for scope, part in [('all',paired),('exclude_logistic_train_and_calibration',
        paired.loc[paired.frozen_cycle_role.eq('not_in_original_fit_or_calibration')])]:
        for (model,policy,label), chunk in part.groupby(['model_id','policy','label'],sort=False):
            valid = chunk.loc[chunk.monitor_available & chunk.old_score.notna()]
            alarm, old_alarm = valid.alert, valid.old_alarm.astype(bool)
            records.append({'model_id':model,'policy':policy,'label':label,'reference':FROZEN_LOGISTIC,'scope':scope,
                'cycle_n':len(chunk),'matched_cycle_n':len(valid),'forecast_alert_n':int(alarm.sum()),
                'logistic_alert_n':int(old_alarm.sum()),'both_alert_n':int((alarm & old_alarm).sum()),
                'forecast_only_alert_n':int((alarm & ~old_alarm).sum()),'logistic_only_alert_n':int((~alarm & old_alarm).sum()),
                'median_lead_from_no3_end_if_both_s':valid.loc[alarm & old_alarm,'seconds_before_end'].median()})
    write_csv(records,'FROZEN_LOGISTIC_CYCLE_COMPARISON',root)
    return {'frozen_logistic_reference_file':str(source),'frozen_logistic_reference_sha256':original_sha,
            'frozen_logistic_bundle_sha256':bundle_sha,
            'frozen_logistic_threshold':float(old.old_threshold.iloc[0]),'unique_reference_cycles':len(old)}


def baseline_parity(native, root):
    source = V1_MULTI/'metrics'/'FIRST_ALERT_TIMES.csv'
    legacy = pd.read_csv(source,usecols=['cycle_id','model_id','policy','alert_before_end','threshold'],low_memory=False)
    archive_e = json.loads((V1_ELECTRIC/'FROZEN_FORECAST_CANDIDATES.json').read_text(encoding='utf-8'))
    archive_e_id = next(s['model_id'] for s in archive_e['forecast_candidates'] if s['input_group']=='F4')
    old_e = pd.read_csv(V1_ELECTRIC/'metrics'/'FIRST_ALERT_TIMES.csv',
        usecols=['cycle_id','model_id','early_alert','residual_threshold'])
    old_e = old_e.loc[old_e.model_id.eq(archive_e_id)].rename(columns={'early_alert':'alert_before_end','residual_threshold':'threshold'})
    legacy = pd.concat([legacy,old_e.assign(policy='E')],ignore_index=True)
    mapping = {
        'v1_e_f4_tree':archive_e_id,'v1_mt_f2_tree':'hist_F2_L2_h0p5_O3_a100_s7181',
        'persistence_L2_h0.5_w0.2':'persistence','template_L2_h0.5_w0.2':'template',
        DIRECT:'direct_vibration_recent_0p2s'}
    records = []
    for (model,policy), rows in native.loc[native.model_id.isin(mapping)].groupby(['model_id','policy'],sort=False):
        old_policy = 'CURRENT_RMS_PEAK' if model==DIRECT else policy
        old = legacy.loc[legacy.model_id.eq(mapping[model]) & legacy.policy.eq(old_policy)]
        if old.empty:
            continue
        assert set(rows.cycle_id) == set(old.cycle_id)
        both = rows.merge(old[['cycle_id','alert_before_end','threshold']],on='cycle_id',suffixes=('','_v1'),validate='one_to_one')
        for label, group in both.groupby('label',sort=False):
            records.append({'model_id':model,'policy':policy,'label':label,'cycle_n':len(group),
                'replay_alerts':int(group.alert.sum()), 'v1_alerts':int(group.alert_before_end.sum()),
                'discordant_cycles':int(group.alert.ne(group.alert_before_end).sum()),
                'replay_threshold':group.threshold.iloc[0],'v1_threshold':group.threshold_v1.iloc[0]})
    table = pd.DataFrame(records)
    write_csv(table,'MONITOR_V1_PARITY',root)
    return {'v1_parity_rows':len(table),'v1_parity_discordant_cycles_sum':int(table.discordant_cycles.sum())}


def select_specs(root,model_ids):
    specs = run_specs(root,model_ids) if model_ids else []
    if len(specs)>3 or any(MONITOR_H not in key_for(s)[1] for s in specs):
        raise ValueError('Select at most three trained candidates that predict the fixed monitoring horizon')
    archived = [s for s in archived_specs(root) if s['id'] in ('v1_e_f4_tree','v1_mt_f2_tree')]
    if len(archived) != 2:
        raise FileNotFoundError('Both archived V1 monitor baselines are required')
    return specs + archived + basic_specs([]) + [{'id':DIRECT,'family':DIRECT}]


def calibrate(root, model_ids):
    data = Data(root)
    specs = select_specs(root,model_ids)
    seal = {'selection_models':list(model_ids),'horizon_s':MONITOR_H,'evaluation_supplied':False,
            'scale':'absolute normal Development residual q90(higher) and positive floors',
            'threshold':'normal Calibration cycle maximum q95(higher), strict >',
            'common_clock':'overlapping .2-second arrival slots, anchored to command send-completion per cycle. Threshold refit per pair on the same rule in Calibration.',
            'rules':{},'paired_rules':[]}
    by_policy = {}
    expected = set(pd.read_csv(root/'SPLIT_MANIFEST.csv', usecols=['role','cycle_id']).query('role == "calibration"').cycle_id)
    for spec in specs:
        is_direct = spec['id']==DIRECT
        parts = event_components(data,spec,'development')
        scales = fit_scales(parts,is_direct)
        calib = event_components(data,spec,'calibration')
        assert calib[0].role.eq('calibration').all() and calib[0].label.eq('normal').all()
        assert set(calib[0].cycle_id).issubset(expected)
        rules = seal['rules'][spec['id']] = {'scales':[float(x) if np.isfinite(x) else None for x in scales], 'policies':{}}
        for policy, indices in ({'RMS_PEAK':[0,1]} if is_direct else POLICIES).items():
            cols = [int(i) for i in indices if np.isfinite(scales[i])]
            if not cols:
                continue
            arrivals = policy_events(calib,cols,scales,is_direct)
            limit, diag = threshold(arrivals,expected,'native')
            by_policy[(spec['id'],policy)] = arrivals
            rules['policies'][policy] = {'columns':cols,**diag}
    requested = set(model_ids)
    # Compare same sensor policy against each baseline; direct vibration can be paired with any policy.
    for (model,policy), events in by_policy.items():
        if model not in requested and model!='v1_mt_f2_tree':
            continue
        refs = [(other,other_policy) for other,other_policy in by_policy if other!=model and other not in requested
                and (policy==other_policy or other==DIRECT)]
        for reference, ref_policy in refs:
            left,right = common_time(events,by_policy[(reference,ref_policy)])
            if left.empty or right.empty:
                continue
            l_rule = threshold(left,expected,'matched')[1]
            r_rule = threshold(right,expected,'matched')[1]
            assert l_rule['calibration_cycles_valid'] == r_rule['calibration_cycles_valid']
            seal['paired_rules'].append({'model_id':model,'policy':policy,'reference':reference,'reference_policy':ref_policy,
                'left':l_rule,'right':r_rule})
    filename = Path(root)/'metrics'/'FROZEN_MONITOR_RULES.json'
    filename.parent.mkdir(parents=True,exist_ok=True)
    filename.write_text(json.dumps(seal,indent=2)+'\n',encoding='utf-8')
    return seal,specs


def paired_monitor(cycles, arr, sealed):
    records = []
    for item in sealed['paired_rules']:
        model,policy,reference,ref_policy = [item[k] for k in ('model_id','policy','reference','reference_policy')]
        ltag,rtag = (model,policy),(reference,ref_policy)
        left,right = common_time(arr[ltag],arr[rtag])
        left = to_cycles(cycles,left,item['left']['threshold'],{'model_id':model,'policy':policy,'scope':'matched'})
        right = to_cycles(cycles,right,item['right']['threshold'],{'model_id':reference,'policy':ref_policy,'scope':'matched'})
        dims = ['model_id','policy','scope','label']
        own = pd.concat([summarize(left,dims),summarize(left,dims+['start_pose'])],ignore_index=True)
        ref = pd.concat([summarize(right,dims),summarize(right,dims+['start_pose'])],ignore_index=True)
        assert own.covered_s.equals(ref.covered_s)
        table = own.merge(ref,on=['scope','label','start_pose'],suffixes=('','_reference'),validate='one_to_one')
        table['threshold'] = item['left']['threshold']
        table['threshold_reference'] = item['right']['threshold']
        table['calibration_common_cycles'] = item['left']['calibration_cycles_valid']
        delta = []
        for item in table.itertuples(index=False):
            label = left.label.eq(item.label)
            if pd.notna(item.start_pose):
                label &= left.start_pose.eq(item.start_pose)
            la,ra = left.loc[label],right.loc[label]
            assert la.cycle_id.equals(ra.cycle_id)
            both = la.alert & ra.alert
            delta.append((int(both.sum()),int((la.alert & ~ra.alert).sum()),int((~la.alert & ra.alert).sum()),
                          (ra.loc[both,'first_alert_arrival_s']-la.loc[both,'first_alert_arrival_s']).median()))
        delta_table = pd.DataFrame(delta,columns=['both_alert_n','own_only_alert_n','reference_only_alert_n','median_seconds_earlier_if_both'])
        table = pd.concat([table,delta_table],axis=1)
        records.append(table)
    return records


def run(root=ROOT, models=(), calibration_only=False, use_seal=False):
    root = Path(root)
    sealed_path = root/'metrics'/'FROZEN_MONITOR_RULES.json'
    if use_seal:
        saved = json.loads(sealed_path.read_text(encoding='utf-8'))
        if set(saved['selection_models']) != set(models) or saved['evaluation_supplied']:
            raise ValueError('Sealed monitor candidates or calibration scope mismatch')
        specs = select_specs(root,models)
        if set(saved['rules']) != {spec['id'] for spec in specs}:
            raise ValueError('Sealed monitor baseline models differ')
    else:
        seal,specs = calibrate(root,models)
    if calibration_only:
        return seal
    initial_seal = sealed_path.read_bytes()
    seal = json.loads(initial_seal)
    data = Data(root)
    cycles = pd.read_csv(root/'SPLIT_MANIFEST.csv',low_memory=False)
    cycles = cycles.loc[cycles.role.eq('evaluation')].reset_index(drop=True)
    assert cycles.cycle_id.is_unique
    outputs,arrivals,thresholds = [],{},[]
    for spec in specs:
        model = spec['id']
        rule = seal['rules'][model]
        scales = np.array([np.nan if s is None else s for s in rule['scales']])
        parts = event_components(data,spec,'evaluation')
        assert set(parts[0].cycle_id).issubset(set(cycles.cycle_id))
        for policy,settings in rule['policies'].items():
            events = policy_events(parts,settings['columns'],scales,model==DIRECT)
            arrivals[model,policy] = events
            tag = {'model_id':model,'policy':policy,'scope':'native'}
            outputs.append(to_cycles(cycles,events,settings['threshold'],tag))
            thresholds.append(dict(tag,**settings))
    comparisons = paired_monitor(cycles,arrivals,seal)
    native = metric_tables(root,outputs,comparisons,thresholds)
    proof = reference_logistic(native,root)
    proof.update(baseline_parity(native,root))
    if sealed_path.read_bytes() != initial_seal:
        raise ValueError('Calibration changed during evaluation')
    proof.update(evaluation_cycles=len(cycles), policies=len(outputs), matched_pairs=len(comparisons),
                 calibration_sha256=hashlib.sha256(initial_seal).hexdigest(),
                 thresholds_set_before_evaluation=True, native_threshold_source='normal calibration only',
                 common_coverage_threshold_recalibration=True, actual_availability_clock_enforced=True,
                 no_contact_delay_without_observed_event_clock=True)
    (root/'metrics'/'MONITOR_VERIFICATION.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    return native


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models',nargs='*',default=[])
    parser.add_argument('--calibration-only',action='store_true')
    parser.add_argument('--use-seal',action='store_true')
    parser.add_argument('--root',type=Path,default=ROOT)
    args = parser.parse_args()
    if args.calibration_only and args.use_seal:
        parser.error('--calibration-only and --use-seal are mutually exclusive')
    run(args.root,args.models,args.calibration_only,args.use_seal)


if __name__ == '__main__':
    main()
