"""Fit, save, verify and replay causal forecast bundles in a local experiment."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from time import perf_counter

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import joblib
import numpy as np
import psutil
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from models import Forecaster, OUTPUT_GROUPS, TARGET_GROUPS, architecture


ROOT = Path(__file__).resolve().parent
LOADED_CODE_SHA256 = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in ('train.py', 'models.py', 'data.py')}
NEURAL = {'gru16', 'gru', 'tcn', 'transformer', 'patch'}
GROUP_NAMES = ('E', 'P', 'RAW', 'STAT')
FLOORS = np.array([1e-4, 1e-4, *([.1] * 6), *([1e-5] * 6)])
SETUP = {'epochs': 32, 'patience': 5, 'batch_size': 256, 'lr': .001,
         'weight_decay': .0001, 'grad_clip': 1., 'min_improvement': 1e-4}


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def normalize_spec(spec):
    merged = dict(group='F4', L=2., horizons=[.5], w=.2, output='O3', size='base', seed=7181,
                  device='cuda', precision='auto', **SETUP)
    merged.update(spec)
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', merged['id']) or merged['id'] in ('.', '..'):
        raise ValueError('Invalid run identifier')
    if merged['family'] not in NEURAL | {'ridge', 'hgb', 'xgb'}:
        raise ValueError('Unknown model family')
    if merged['size'] not in ('base', 'large') or merged['output'] not in OUTPUT_GROUPS:
        raise ValueError('Invalid capacity or output selection')
    return merged


def file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ResourceMonitor:
    """Track process resident RAM while fitting and reloading a single bundle."""
    def __init__(self, gpu=False):
        self.nvml = None
        if gpu:
            from gpu_memory import DeviceMemory

            try:
                self.nvml = DeviceMemory()
                self.device_baseline = self.nvml.used_mib()
                self.device_peak = self.device_baseline
            except (OSError, RuntimeError):
                self.nvml = None

    def __enter__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.event = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        while not self.event.wait(.1):
            self.peak = max(self.peak, self.process.memory_info().rss)
            if self.nvml is not None:
                self.device_peak = max(self.device_peak, self.nvml.used_mib())

    def __exit__(self, *args):
        self.event.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def train_scalers(x, y):
    flat = x.reshape(-1, x.shape[-1])
    fill = [float(np.median(col[np.isfinite(col)])) if np.isfinite(col).any() else 0. for col in flat.T]
    scaler = StandardScaler().fit(np.where(np.isfinite(flat), flat, fill).astype(np.float32))
    mean = np.nanmean(y.astype(np.float64), axis=0)
    std = np.maximum(np.nanstd(y.astype(np.float64), axis=0), FLOORS)
    # A wholly unavailable target is masked throughout and has a neutral restoration scale.
    return dict(x_fill=fill, x_mean=scaler.mean_.tolist(), x_std=scaler.scale_.tolist(),
                y_mean=np.where(np.isfinite(mean), mean, 0.).tolist(),
                y_std=np.where(np.isfinite(std), std, 1.).tolist())


def scale_x(x, stats):
    mean, scale, fill = (np.asarray(stats[key], dtype=np.float32) for key in ('x_mean', 'x_std', 'x_fill'))
    return (np.where(np.isfinite(x), x, fill).astype(np.float32) - mean) / scale


def scale_y(y, stats):
    return ((y - np.array(stats['y_mean'])) / np.array(stats['y_std'])).astype(np.float32)


def output_slices(output):
    used = [TARGET_GROUPS[group] for group in OUTPUT_GROUPS[output]]
    counts = np.cumsum([0, *(len(group) for group in used)])
    return [idx for group in used for idx in group], [slice(a, b) for a, b in zip(counts[:-1], counts[1:])]


def cycle_weights(metadata):
    sizes = metadata.groupby('cycle_id')['cycle_id'].transform('size').to_numpy(dtype=np.float32)
    weights = 1. / sizes
    return weights / weights.mean()


def group_loss(pred, values, mask, weights, groups):
    """Average valid scaled squared errors within and then across sensor groups."""
    weight = mask * weights[:, None, None]
    sq = (pred.float() - values).square() * weight
    sums = torch.stack([sq[:, :, group].sum() for group in groups])
    counts = torch.stack([weight[:, :, group].sum() for group in groups])
    per_group = sums / counts.clamp_min(1e-12)
    total = (per_group * counts.gt(0)).sum() / counts.gt(0).sum().clamp_min(1)
    return total, sums.detach(), counts.detach()


def dev_score(pred, actual, output, weights):
    keep = [TARGET_GROUPS[group] for group in OUTPUT_GROUPS[output]]
    errors = (pred - actual) ** 2
    return float(np.mean([np.nansum(errors[:, :, g] * weights[:, None, None]) /
                          np.sum(np.isfinite(errors[:, :, g]) * weights[:, None, None]) for g in keep]))


def save_weights(path, data):
    temp = path.with_suffix('.tmp')
    torch.save(data, temp)
    temp.replace(path)


def configure_torch_precision():
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')


def fit_neural(run, spec, detail, x, y, weight, dx, dy, dweight):
    device = torch.device(spec['device'])
    torch.manual_seed(spec['seed'])
    configure_torch_precision()
    model = Forecaster(detail).to(device)
    baseline = spec['family'] == 'gru16'
    if baseline:
        epochs, patience, batch_size = 16, 3, 512
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        weight, dweight = np.ones_like(weight), np.ones_like(dweight)
    else:
        epochs, patience, batch_size = spec['epochs'], spec['patience'], spec['batch_size']
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec['lr'], weight_decay=spec['weight_decay'])
    mixed = device.type == 'cuda' and spec['family'] in {'transformer', 'patch', 'tcn'} and spec['precision'] != 'fp32'
    indices, groups = output_slices(spec['output'])

    def prepare(features, actual, row_weight):
        values = torch.as_tensor(actual[:, :, indices], device=device)
        return (torch.as_tensor(features, device=device), torch.nan_to_num(values),
                torch.isfinite(values), torch.as_tensor(row_weight, device=device))

    inputs, targets, valid, row_weight = prepare(x, y, weight)
    dev_inputs, dev_targets, dev_valid, dev_weights = prepare(dx, dy, dweight)
    resume = run / 'RESUME.pt'
    records, best, best_epoch, stale, first_epoch, previous = [], float('inf'), 0, 0, 1, 0.
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=True)
        if state['spec'] != spec:
            raise ValueError(f'Resume spec changed for {spec["id"]}')
        model.load_state_dict(state['last'])
        optimizer.load_state_dict(state['optimizer'])
        records, best, best_epoch, stale = (state[key] for key in ('records', 'best', 'best_epoch', 'stale'))
        first_epoch, previous = state['epoch'] + 1, state['elapsed_s']
    start = perf_counter()

    for epoch in range(first_epoch, epochs + 1) if stale < patience else ():
        epoch_started = perf_counter()
        model.train()
        torch.manual_seed(spec['seed'] + epoch)
        ordering = torch.randperm(len(x), device=device)
        train_sums = torch.zeros(len(groups), device=device)
        train_counts = torch.zeros_like(train_sums)
        for batch in ordering.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=mixed):
                estimated = model(inputs[batch])
            loss, sums, counts = group_loss(estimated, targets[batch], valid[batch], row_weight[batch], groups)
            loss.backward()
            if not baseline:
                nn.utils.clip_grad_norm_(model.parameters(), spec['grad_clip'])
            optimizer.step()
            train_sums += sums
            train_counts += counts
        model.eval()
        dev_sums = torch.zeros_like(train_sums)
        dev_counts = torch.zeros_like(train_sums)
        # FP32 validation and inference use the same saved weights.
        with torch.inference_mode():
            for batch in torch.arange(len(dx), device=device).split(batch_size):
                _, sums, counts = group_loss(model(dev_inputs[batch]), dev_targets[batch], dev_valid[batch],
                                              dev_weights[batch], groups)
                dev_sums += sums
                dev_counts += counts
        train_groups = (train_sums / train_counts).cpu().tolist()
        dev_groups = (dev_sums / dev_counts).cpu().tolist()
        dev_total = float(np.mean(dev_groups))
        if not np.isfinite(dev_total):
            raise FloatingPointError('Development loss is not finite')
        for group, a, b in zip(OUTPUT_GROUPS[spec['output']], train_groups, dev_groups):
            records.append(dict(epoch=epoch, target_group=GROUP_NAMES[group], train_loss=a, dev_loss=b,
                                dev_loss_all=dev_total, elapsed_s=perf_counter() - epoch_started))
        if dev_total < best - spec['min_improvement']:
            best, best_epoch, stale = dev_total, epoch, 0
            save_weights(run / 'model.pt', model.state_dict())
        else:
            stale += 1
        if epoch % 5 == 0 or epoch == epochs or stale == patience:
            print(f'{spec["id"]} epoch {epoch}, dev={dev_total:.6f}, best={best:.6f}', flush=True)
        save_weights(resume, dict(spec=spec, last=model.state_dict(), optimizer=optimizer.state_dict(), records=records,
                                  best=best, best_epoch=best_epoch, stale=stale, epoch=epoch,
                                  elapsed_s=previous + perf_counter() - start))
        if stale == patience:
            break
    with (run / 'EPOCHS.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    model.load_state_dict(torch.load(run / 'model.pt', map_location=device, weights_only=True))
    model.eval()
    return model, dict(fit_s=previous + perf_counter() - start, dev_loss=best, selected_epoch=best_epoch,
                       epochs_run=records[-1]['epoch'], parameter_count=sum(p.numel() for p in model.parameters()),
                       estimator_count=1, mixed_precision='bfloat16' if mixed else 'float32', fit_device=str(device),
                       optimization={'batch_size': batch_size, 'epochs': epochs, 'patience': patience,
                                     'optimizer': type(optimizer).__name__, 'dev_precision': 'float32 with TF32 enabled',
                                     'train_cycle_weight': 'equal_total_per_cycle' if not baseline else 'legacy_equal_window'})


def xgb_parameters(spec):
    large = spec['size'] == 'large'
    return dict(max_depth=8 if large else 5, n_estimators=900 if large else 500,
                learning_rate=.045 if large else .06, min_child_weight=6, reg_lambda=3.,
                subsample=.9, colsample_bytree=.9, early_stopping_rounds=25,
                tree_method='hist', device=spec['device'], n_jobs=4, random_state=spec['seed'], max_bin=128)


def fit_independent(run, spec, x, y, weight, dx, dy, dweight):
    from xgboost import XGBRegressor

    started = perf_counter()
    family = spec['family']
    flat, dev = x.reshape(len(x), -1), dx.reshape(len(dx), -1)
    indices, _ = output_slices(spec['output'])
    bundle, trees, tree_limit, estimator_seconds = [], [], [], []
    resumed = run / 'TREE_RESUME.joblib'
    if resumed.exists():
        previous = joblib.load(resumed)
        if previous['spec'] != spec:
            raise ValueError(f'Resume spec changed for {spec["id"]}')
        bundle, trees, tree_limit, estimator_seconds = (previous[key] for key in
                                                         ('bundle', 'trees', 'tree_limit', 'estimator_seconds'))
    for h, target in [(h, target) for h in range(y.shape[1]) for target in indices][len(bundle):]:
        valid, dv = np.isfinite(y[:, h, target]), np.isfinite(dy[:, h, target])
        if valid.sum() < 50 or dv.sum() < 10:
            raise ValueError(f'Insufficient valid targets for horizon {h}, target {target}')
        start = perf_counter()
        if family == 'ridge':
            estimator = Ridge(alpha=spec.get('ridge_alpha', 25), solver='cholesky').fit(flat[valid], y[valid, h, target])
            count, cap = 0, 0
        elif family == 'hgb':
            estimator = HistGradientBoostingRegressor(max_iter=65, max_leaf_nodes=15, max_bins=64,
                min_samples_leaf=45, early_stopping=False, l2_regularization=2., random_state=spec['seed'])
            estimator.fit(flat[valid], y[valid, h, target])
            count, cap = estimator.n_iter_, estimator.max_iter
        else:
            estimator = XGBRegressor(**xgb_parameters(spec))
            estimator.fit(flat[valid], y[valid, h, target], sample_weight=weight[valid],
                          eval_set=[(dev[dv], dy[dv, h, target])], sample_weight_eval_set=[dweight[dv]], verbose=False)
            count, cap = estimator.best_iteration + 1, estimator.get_booster().num_boosted_rounds()
            estimator.set_params(device='cpu')
        estimator_seconds.append(perf_counter() - start)
        bundle.append(estimator)
        trees.append(count)
        tree_limit.append(cap)
        temp = resumed.with_suffix('.tmp')
        joblib.dump(dict(spec=spec, bundle=bundle, trees=trees, tree_limit=tree_limit,
                         estimator_seconds=estimator_seconds), temp, compress=0)
        temp.replace(resumed)
        if family == 'xgb':
            print(f'{spec["id"]} target {h}:{target}, {count} best trees, {estimator_seconds[-1]:.2f}s', flush=True)
    joblib.dump(bundle, run / 'model.joblib', compress=3)
    return bundle, dict(fit_s=sum(estimator_seconds), bundle_prepare_s=perf_counter() - started,
                        epochs_run=0, parameter_count=0, estimator_count=len(bundle), trees=trees,
                        built_trees=tree_limit, estimator_seconds=estimator_seconds,
                        mixed_precision='float32', fit_device=spec['device'] if family == 'xgb' else 'cpu',
                        ridge_alpha=spec.get('ridge_alpha', 25) if family == 'ridge' else None,
                        xgb_parameters=xgb_parameters(spec) if family == 'xgb' else None)


class Predictor:
    def __init__(self, spec, stats, fitted):
        self.spec, self.stats, self.fitted = spec, stats, fitted
        self.indices = output_slices(spec['output'])[0]
        self.is_neural = spec['family'] in NEURAL
        if self.is_neural:
            self.device = next(fitted.parameters()).device

    def sync(self):
        if self.is_neural and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def __call__(self, raw):
        return self.predict_scaled(scale_x(raw, self.stats))

    def predict_scaled(self, x):
        horizons, ids = len(self.spec['horizons']), self.indices
        if self.is_neural:
            with torch.inference_mode():
                z = np.concatenate([self.fitted(chunk.to(self.device)).cpu().numpy()
                                    for chunk in torch.from_numpy(x).split(512)])
        else:
            flat = x.reshape(len(x), -1)
            z = np.column_stack([model.predict(flat) for model in self.fitted]).reshape(len(x), horizons, len(ids))
        result = np.full((len(x), horizons, 14), np.nan, dtype=np.float32)
        result[:, :, ids] = z * np.asarray(self.stats['y_std'])[:, ids] + np.asarray(self.stats['y_mean'])[:, ids]
        return result


def load_predictor(run, device=None):
    run = Path(run)
    stored = json.loads((run / 'RESULT.json').read_text(encoding='utf-8'))
    spec, stats, detail = (stored[k] for k in ('spec', 'scalers', 'architecture'))
    if spec['family'] in NEURAL:
        configure_torch_precision()
        actual_device = device or spec['device']
        fitted = Forecaster(detail).to(actual_device)
        fitted.load_state_dict(torch.load(run / 'model.pt', weights_only=True, map_location=actual_device))
        fitted.eval()
    else:
        fitted = joblib.load(run / 'model.joblib')
    return Predictor(spec, stats, fitted)


def timing(predictor, raw):
    scaled = scale_x(raw[:64], predictor.stats)
    for _ in range(6):
        predictor.predict_scaled(scaled[:1])
        predictor.predict_scaled(scaled)
    predictor.sync()
    samples = []
    for _ in range(35):
        t = perf_counter()
        predictor.predict_scaled(scaled[:1])
        predictor.sync()
        samples.append(1000 * (perf_counter() - t))
    start = perf_counter()
    for _ in range(12):
        predictor.predict_scaled(scaled)
    predictor.sync()
    batch64_s = (perf_counter() - start) / 12
    return dict(batch1_p50_ms=float(np.percentile(samples, 50)), batch1_p95_ms=float(np.percentile(samples, 95)),
                batch64_seconds=batch64_s, batch64_rows_per_s=len(scaled) / batch64_s,
                timing_transfer='model forward including host-device copies; scaler excluded',
                inference_device=str(predictor.device) if predictor.is_neural else 'cpu')


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).view('uint8')).hexdigest()


def run_one(spec, data):
    from data import CHANNELS, GROUPS, TARGETS

    run = ROOT / 'runs' / spec['id']
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'RESULT.json').exists() and (run / 'PREDICTIONS_development.npy').exists():
        result = json.loads((run / 'RESULT.json').read_text(encoding='utf-8'))
        if result['spec'] != spec:
            raise ValueError(f'Existing complete run has a different spec: {spec["id"]}')
        print(f'{spec["id"]}: cached', flush=True)
        return
    started = perf_counter()
    is_neural = spec['family'] in NEURAL
    use_cuda = spec['device'].startswith('cuda') and (is_neural or spec['family'] == 'xgb')
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    with ResourceMonitor(gpu=use_cuda) as resource:
        x, y, train_meta = data.get('train', spec['L'], spec['horizons'], spec['group'], spec['w'])
        dx, dy, dev_meta = data.get('development', spec['L'], spec['horizons'], spec['group'], spec['w'])
        if set(train_meta.cycle_id) & set(dev_meta.cycle_id):
            raise ValueError('Training/development cycle overlap')
        weight, dev_weight = cycle_weights(train_meta), cycle_weights(dev_meta)
        stats = train_scalers(x, y)
        x = scale_x(x, stats)
        detail = architecture(spec, x.shape[-1])
        print(f'{spec["id"]}: {len(x)} Train, {len(dx)} Development, {detail}', flush=True)
        method = fit_neural if is_neural else fit_independent
        with threadpool_limits(limits=4):
            fitted, metrics = method(run, spec, detail, x, scale_y(y, stats), weight,
                                     scale_x(dx, stats), scale_y(dy, stats), dev_weight) if is_neural else method(
                                         run, spec, x, scale_y(y, stats), weight,
                                         scale_x(dx, stats), scale_y(dy, stats), dev_weight)
        predictor = Predictor(spec, stats, fitted)
        with threadpool_limits(limits=4):
            predictions = predictor(dx)
            timing_results = timing(predictor, dx)
        indices, _ = output_slices(spec['output'])
        assert np.isfinite(predictions[:, :, indices]).all()
        model_file = run / ('model.pt' if is_neural else 'model.joblib')
        gpu_peak = torch.cuda.max_memory_allocated() / 2**20 if use_cuda and is_neural else None
        gpu_reserved = torch.cuda.max_memory_reserved() / 2**20 if use_cuda and is_neural else None
        norm_error = (predictions - dy) / np.asarray(stats['y_std'])
        result = dict(spec=spec, architecture=detail, scalers=stats,
                      input_features=[CHANNELS[index] for index in GROUPS[spec['group']]],
                      target_names=list(TARGETS), output_target_indices=indices,
                      train_n=len(x), development_n=len(dx), train_cycle_count=train_meta.cycle_id.nunique(),
                      development_cycle_count=dev_meta.cycle_id.nunique(),
                      train_cycle_ids=train_meta.cycle_id.drop_duplicates().astype(str).tolist(),
                      development_cycle_ids=dev_meta.cycle_id.drop_duplicates().astype(str).tolist(),
                      train_y_sha256=array_digest(y), development_y_sha256=array_digest(dy),
                      code_sha256=LOADED_CODE_SHA256,
                      normalized_dev_mse_by_target=[[float(np.nanmean(norm_error[:, h, j] ** 2)) if j in indices else None
                                                       for j in range(14)] for h in range(len(spec['horizons']))],
                      dev_selection_loss=dev_score(predictions / np.asarray(stats['y_std']),
                                                   dy / np.asarray(stats['y_std']), spec['output'], dev_weight),
                      model_sha256=file_digest(model_file), model_file=model_file.name,
                      model_bytes=model_file.stat().st_size, torch_version=torch.__version__,
                      cuda_runtime=torch.version.cuda, gpu_peak_allocated_mib=gpu_peak,
                      gpu_peak_reserved_mib=gpu_reserved, **metrics, **timing_results)
        # Replay through a new model instance. Batch order must leave each window independent.
        write_json(run / 'RESULT.json', result)
        reloading = perf_counter()
        reloaded = load_predictor(run)
        reloaded.sync()
        result['model_load_s'] = perf_counter() - reloading
        block = dx[:16]
        with threadpool_limits(limits=4):
            expected = predictor(block)
            batch = reloaded(block[::-1])[::-1]
            single = np.concatenate([reloaded(row[None]) for row in block])
        delta, single_delta = (float(np.nanmax(abs(expected - actual))) for actual in (batch, single))
        normalized_delta = float(np.nanmax(abs(expected - single) / np.asarray(stats['y_std'])))
        if not np.allclose(expected, batch, atol=2e-3, rtol=1e-4, equal_nan=True) or normalized_delta > .005:
            raise ValueError(f'Save/reload or inference order mismatch {delta}, {single_delta}, {normalized_delta}')
        result.update(reload_and_reverse_max_abs_delta=delta, batch_vs_single_max_abs_delta=single_delta,
                      batch_vs_single_normalized_max_abs_delta=normalized_delta)
    result.update(process_ram_peak_mib=resource.peak / 2**20, total_run_s=perf_counter() - started)
    if resource.nvml is not None:
        result.update(nvml_global_before_mib=resource.device_baseline,
                      nvml_global_peak_used_mib=resource.device_peak,
                      nvml_global_increase_mib=resource.device_peak - resource.device_baseline,
                      nvml_scope='device total at 100 ms intervals, other processes included')
    dev_prediction_path = run / 'PREDICTIONS_development.npy'
    save_predictions(dev_prediction_path, predictions)
    result['development_prediction_sha256'] = file_digest(dev_prediction_path)
    write_json(run / 'RESULT.json', result)
    for scratch in ('TREE_RESUME.joblib', 'RESUME.pt', 'FAILURE.json'):
        (run / scratch).unlink(missing_ok=True)
    print(f'{spec["id"]}: complete, fit {result["fit_s"]:.2f}s, dev loss {result["dev_selection_loss"]:.6f}', flush=True)


def save_predictions(path, values):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        np.save(stream, values, allow_pickle=False)
    temp.replace(path)


def predict(spec, data, roles):
    run = ROOT / 'runs' / spec['id']
    stored = json.loads((run / 'RESULT.json').read_text(encoding='utf-8'))
    if stored['spec'] != spec:
        raise ValueError(f'Prediction spec changed for {spec["id"]}')
    model = load_predictor(run)
    metadata_path = run / 'ROLE_PREDICTION_INDEX.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
    with threadpool_limits(limits=4):
        for role in roles:
            path = run / f'PREDICTIONS_{role}.npy'
            if path.exists():
                continue
            x, y, rows = data.get(role, spec['L'], spec['horizons'], spec['group'], spec['w'])
            started = perf_counter()
            prediction = model(x)
            model.sync()
            save_predictions(path, prediction)
            elapsed = perf_counter() - started
            metadata[role] = dict(rows=len(rows), cycles=rows.cycle_id.nunique(), target_sha256=array_digest(y),
                                  prediction_sha256=file_digest(path), prediction_s=elapsed,
                                  model_sha256=stored['model_sha256'])
            write_json(metadata_path, metadata)
            print(f'{spec["id"]}: {role} {len(x)} predictions in {elapsed:.2f}s', flush=True)


def smoke():
    from xgboost import XGBRegressor, build_info

    torch.manual_seed(7181)
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    base = torch.rand(64, 20, 27, device=device)
    assert torch.isfinite(base @ base.transpose(-1, -2)).all()
    tests = []
    for family in ('gru', 'tcn', 'transformer', 'patch'):
        spec = normalize_spec({'id': 'synthetic', 'family': family})
        detail = architecture(spec, 27)
        model = Forecaster(detail).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        before = [param.detach().clone() for param in model.parameters()]
        mixed = family != 'gru'
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=mixed):
            outputs = model(base)
        valid = torch.ones_like(outputs, dtype=torch.bool)
        valid[::2, :, 0] = False
        loss, _, _ = group_loss(outputs, torch.zeros_like(outputs, dtype=torch.float32), valid,
                                torch.ones(len(base), device=device), output_slices('O3')[1])
        loss.backward()
        optimizer.step()
        assert any(not a.equal(b) for a, b in zip(before, model.parameters()))
        original = model.eval()(base).detach()
        path = ROOT / 'SMOKE_WEIGHTS.pt'
        save_weights(path, model.state_dict())
        loaded = Forecaster(detail).to(device).eval()
        loaded.load_state_dict(torch.load(path, weights_only=True))
        path.unlink()
        delta = (original - loaded(base)).abs().max().item()
        assert delta == 0
        tests.append(dict(family=family, loss=loss.item(), weight_update=True,
                          reload_max_abs_delta=delta, mixed_precision='bfloat16' if mixed else 'float32'))
    sample = base[:, -1].cpu().numpy()
    reg = XGBRegressor(max_depth=3, n_estimators=12, early_stopping_rounds=2, tree_method='hist', device='cuda')
    reg.fit(sample[:48], sample[:48, 0], eval_set=[(sample[48:], sample[48:, 0])], verbose=False)
    reg.set_params(device='cpu')
    assert np.isfinite(reg.predict(sample[48:])).all()
    write_json(ROOT / 'GPU_SMOKE.json', dict(torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
        compiled_archs=torch.cuda.get_arch_list(), tests=tests, xgboost_build=build_info(),
        xgboost_gpu_best_trees=reg.best_iteration + 1, xgboost_gpu_forward=True))
    print('GPU smoke passed for all four neural families and XGBoost', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('smoke', 'run', 'predict'))
    parser.add_argument('spec', nargs='?', help='Path to a JSON spec, a list, or a JSON object')
    parser.add_argument('--roles', nargs='+', default=('calibration', 'evaluation'))
    parser.add_argument('--only', nargs='+', help='Optional subset of spec IDs')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == 'smoke':
        smoke()
        return
    from data import Data
    specs = json.loads(args.spec if args.spec.startswith(('[', '{')) else Path(args.spec).read_text(encoding='utf-8-sig'))
    specs = [specs] if isinstance(specs, dict) else specs
    d = Data()
    failures = 0
    for raw in specs:
        spec = normalize_spec(raw)
        if args.only and spec['id'] not in args.only:
            continue
        try:
            if args.command == 'run':
                run_one(spec, d)
            else:
                predict(spec, d, args.roles)
        except Exception as exc:
            run = ROOT / 'runs' / spec['id']
            run.mkdir(parents=True, exist_ok=True)
            write_json(run / 'FAILURE.json', dict(spec=spec, command=args.command, error=repr(exc)))
            import traceback
            traceback.print_exc()
            failures += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if failures:
        raise SystemExit(f'{failures} model bundles failed. See FAILURE.json files and execution log.')


if __name__ == '__main__':
    main()
