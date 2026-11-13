"""Local TabPFN-v2 evaluation with fixed train contexts and group holdouts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parent
for key, value in {'TABPFN_DISABLE_TELEMETRY':'1','TABPFN_NO_BROWSER':'1',
                   'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1',
                   'SKB_DATA_DIRECTORY':str(ROOT/'runtime_cache/skrub'),
                   'TABPFN_MODEL_CACHE_DIR':str(ROOT/'runtime_cache/tabpfn'),
                   'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}.items():
    os.environ[key] = value

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import log_loss
import torch
from tabpfn import TabPFNClassifier
from tabpfn.preprocessing.datamodel import FeatureModality

CONFIG = json.loads((ROOT/'EXPERIMENT_CONFIG.json').read_text(encoding='utf-8'))
CHECKPOINT = ROOT/CONFIG['checkpoint']
CLASSES = CONFIG['classes']
PROB_COLUMNS = [f'p_{label}' for label in CLASSES]
ROLES = ('development','calibration','evaluation')
PROB_ATOL = 5e-4
INFERENCE = dict(MIN_UNIQUE_FOR_NUMERICAL_FEATURES=0,ENABLE_GPU_PREPROCESSING=False,
                 TRANSFORM_TEXT=False,TRANSFORM_DATES=False,SUBSAMPLE_SAMPLES=None)


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def json_write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def classifier(path, seed):
    return TabPFNClassifier(device='cuda',model_path=path,n_estimators=CONFIG['ensemble'],
        categorical_features_indices=[],random_state=seed,balance_probabilities=False,
        inference_config=INFERENCE,fit_mode='fit_preprocessors')


def device_seconds(function, *args):
    torch.cuda.synchronize()
    start=perf_counter()
    value=function(*args)
    torch.cuda.synchronize()
    return value,perf_counter()-start


def ordered_prob(model, x):
    assert set(model.classes_)==set(range(4)), model.classes_
    by_class=[list(model.classes_).index(i) for i in range(4)]
    output=np.asarray(model.predict_proba(x),dtype=np.float64)[:,by_class]
    assert output.shape==(len(x),4) and np.isfinite(output).all()
    np.testing.assert_allclose(output.sum(axis=1),1,atol=1e-5)
    return output/output.sum(axis=1,keepdims=True)


def label_ids(labels):
    return labels.map(dict(zip(CLASSES,range(4)))).to_numpy(dtype=np.int64)


def predict(model, values):
    batch=CONFIG['inference_query_batch']
    return np.concatenate([ordered_prob(model,values[i:i+batch]) for i in range(0,len(values),batch)])


def raw_score_table(parts, probabilities, split, schema, seed, method):
    threshold=float(np.quantile(1-probabilities['calibration'][:,0],.95,method='higher'))
    frames=[]
    for role, values in probabilities.items():
        score=1-values[:,0]
        table=parts[role][['cycle_id','label']].assign(split_id=split,schema=schema,seed=seed,method=method,
            role=role,score=score,threshold=threshold,alarm=score>threshold)
        for col,prob in zip(PROB_COLUMNS,values.T):
            table[col]=prob
        frames.append(table)
    return pd.concat(frames,ignore_index=True),threshold


def check_batch(model, values, threshold):
    sample=values[:12]
    batch=ordered_prob(model,sample)
    single=np.concatenate([ordered_prob(model,row[None,:]) for row in sample])
    reverse=ordered_prob(model,sample[::-1].copy())[::-1]
    repeat=ordered_prob(model,sample)
    return dict(batch_vs_single_max=float(np.max(np.abs(batch-single))),
                reverse_max=float(np.max(np.abs(batch-reverse))),repeat_max=float(np.max(np.abs(batch-repeat))),
                batch_vs_single_alarm_mismatch=int(np.count_nonzero((1-batch[:,0]>threshold)!=(1-single[:,0]>threshold))),
                reverse_alarm_mismatch=int(np.count_nonzero((1-batch[:,0]>threshold)!=(1-reverse[:,0]>threshold))),
                tolerance=PROB_ATOL)


def run_case(source, split, schema, seed, mode, smoke=False, repeat_lr=None):
    assert sha(CHECKPOINT)==CONFIG['checkpoint_sha256']
    folder=ROOT/'runs'/split/schema/str(seed)
    folder.mkdir(parents=True,exist_ok=True)
    context_file=ROOT/'contexts'/split/f'{schema}.csv'
    context=pd.read_csv(context_file,float_precision='round_trip')
    columns=json.loads((ROOT/'FEATURE_SCHEMAS.json').read_text(encoding='utf-8'))[schema]
    manifest=pd.read_csv(ROOT/'SPLIT_MANIFEST.csv')
    subset=manifest.loc[manifest.split_id.eq(split)]
    assert subset.loc[subset.role.eq('train'),'cycle_id'].tolist()==context.cycle_id.tolist()
    features=pd.read_csv(source/'data/FEATURES.csv').set_index('cycle_id')
    parts={role:subset.loc[subset.role.eq(role)].copy() for role in ROLES}
    x={role:features.loc[rows.cycle_id,columns].to_numpy(dtype=np.float64) for role,rows in parts.items()}
    train=context[columns].to_numpy(dtype=np.float64)
    y=label_ids(context.label)
    dev=label_ids(parts['development'].label)
    assert all(np.isfinite(arr).all() for arr in (train,*x.values()))
    np.testing.assert_array_equal(train,features.loc[context.cycle_id,columns])
    assert all(set(context.cycle_id).isdisjoint(rows.cycle_id) for rows in parts.values())
    if smoke:
        from finetune_local import fit_local
        rng=np.random.default_rng(103)
        synthetic=rng.normal(size=(96,24))
        labels=(np.arange(96)%4).astype(np.int64)
        torch.cuda.reset_peak_memory_stats()
        fitted, elapsed=device_seconds(classifier(CHECKPOINT,seed).fit,synthetic[:64],labels[:64])
        p=ordered_prob(fitted,synthetic[64:70])
        # Validation intentionally lacks class 1, matching the outer Development structure.
        val=(labels[64:]!=1)
        tested=fit_local(CHECKPOINT,synthetic[:64],labels[:64],synthetic[64:][val],labels[64:][val],
                         seed,1e-5,ROOT/'smoke_ft',n_estimators=CONFIG['ensemble'],epochs=1)
        q=ordered_prob(tested,synthetic[64:70])
        json_write(ROOT/'SMOKE.json',dict(context_fit_s=elapsed,gpu=torch.cuda.get_device_name(),
            max_allocated= torch.cuda.max_memory_allocated(),p0_shape=list(p.shape),p1_shape=list(q.shape),
            cuda=torch.version.cuda,torch=torch.__version__,validation_missing_class_test=True))
        return
    if torch.cuda.mem_get_info()[0] < 3*2**30:
        raise RuntimeError('Insufficient free CUDA memory for TabPFN')
    torch.cuda.reset_peak_memory_stats()
    if mode=='P0':
        model_path=CHECKPOINT
        t=perf_counter()
        model=classifier(model_path,seed)
        construction_s=perf_counter()-t
        model,fit_s=device_seconds(model.fit,train,y)
        _,first_s=device_seconds(ordered_prob,model,x['development'][:1])
        pdev, dev_s=device_seconds(predict,model,x['development'])
        val_loss=log_loss(dev,pdev,labels=np.arange(4))
        json_write(folder/'P0_selection.json',dict(dev_loss=val_loss,context_sha256=sha(context_file)))
        info=dict(context_fit_s=fit_s,construction_s=construction_s,first_prediction_ms=first_s*1000,dev_predict_s=dev_s,
                  ft_s=0.0,steps=0,selected_epoch=0,dev_loss=val_loss)
    else:
        from finetune_local import fit_local
        p0=json.loads((folder/'P0_selection.json').read_text())
        base_loss=p0['dev_loss']
        choices=[]
        rates=CONFIG['fine_tuning_lr'] if repeat_lr is None else [repeat_lr]
        for rate in rates:
            candidate_dir=folder/f'lr_{rate:g}'
            start=perf_counter()
            candidate=fit_local(CHECKPOINT,train,y,x['development'],dev,seed,rate,candidate_dir,
                                n_estimators=CONFIG['ensemble'],epochs=CONFIG['fine_tuning_max_epochs'])
            val=log_loss(dev,predict(candidate,x['development']),labels=np.arange(4))
            ft_info=json.loads((candidate_dir/'FIT.json').read_text())
            choices.append((val,rate,ft_info,perf_counter()-start))
            del candidate
            torch.cuda.empty_cache()
        best_loss,rate,ft_info,_=min(choices,key=lambda c:(c[0],c[1]))
        improve=best_loss < base_loss-1e-8 and ft_info['selected_epoch']>0
        model_path=(folder/f'lr_{rate:g}'/ft_info['checkpoint']) if improve else CHECKPOINT
        t=perf_counter()
        model=classifier(model_path,seed)
        construction_s=perf_counter()-t
        model,fit_s=device_seconds(model.fit,train,y)
        _,first_s=device_seconds(ordered_prob,model,x['development'][:1])
        pdev,dev_s=device_seconds(predict,model,x['development'])
        val_loss=log_loss(dev,pdev,labels=np.arange(4))
        np.testing.assert_allclose(val_loss,best_loss if improve else base_loss,atol=PROB_ATOL,rtol=0)
        info=dict(context_fit_s=fit_s,construction_s=construction_s,first_prediction_ms=first_s*1000,dev_predict_s=dev_s,
                  ft_s=sum(c[3] for c in choices),steps=sum(c[2]['optimizer_steps'] for c in choices),
                  selected_epoch=ft_info['selected_epoch'] if improve else 0,dev_loss=val_loss,
                  lr=rate if improve else None,attempts=[{**c[2],'runner_dev_loss':c[0],'lr':c[1]} for c in choices])
        json_write(folder/'P1_selection.json',info)
    begin=perf_counter()
    cal=predict(model,x['calibration'])
    evaluation, evaluation_s=device_seconds(predict,model,x['evaluation'])
    scores,threshold=raw_score_table(parts,dict(development=pdev,calibration=cal,evaluation=evaluation),
                                    split,schema,seed,mode)
    scores.to_csv(folder/f'{mode}_scores.csv',index=False)
    diagnostics=check_batch(model,x['evaluation'],threshold)
    if any(diagnostics[k]>PROB_ATOL for k in ('batch_vs_single_max','reverse_max','repeat_max')):
        print(f'Query-layout difference above tolerance: {diagnostics}',flush=True)
    sample=x['evaluation'][:64]
    _, batch_s=device_seconds(ordered_prob,model,sample)
    single=[device_seconds(ordered_prob,model,row[None,:])[1] for row in sample[:12]]
    model_file=Path(model_path)
    details=dict(split_id=split,schema=schema,seed=seed,method=mode,context_n=len(train),feature_count=len(columns),
        context_file=context_file.relative_to(ROOT).as_posix(),context_sha256=sha(context_file),
        checkpoint=model_file.relative_to(ROOT).as_posix(),checkpoint_sha256=sha(model_file),threshold=threshold,
        model_parameters=sum(p.numel() for backbone in model.models_ for p in backbone.parameters()),
        trainable_parameters=ft_info.get('trainable_parameters') if mode=='P1' else 0,
        numeric_indices=model.inferred_feature_schema_.indices_for(FeatureModality.NUMERICAL),
        artifact_bytes=model_file.stat().st_size+context_file.stat().st_size,
        eval_rows_per_s=len(evaluation)/evaluation_s,batch64_rows_per_s=len(sample)/batch_s,
        p50_ms=float(np.median(single))*1000,p95_ms=float(np.quantile(single,.95))*1000,
        gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20,gpu_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        rss_mib=psutil.Process().memory_info().rss/2**20,score_and_checks_s=perf_counter()-begin,
        **info,**diagnostics)
    assert details['numeric_indices']==list(range(len(columns))),details['numeric_indices']
    json_write(folder/f'{mode}_details.json',details)
    print(split,schema,seed,mode,details['dev_loss'],details['selected_epoch'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('split',choices=CONFIG['splits'])
    p.add_argument('schema',choices=CONFIG['schemas'])
    p.add_argument('mode',choices=['P0','P1'])
    p.add_argument('--seed',type=int,default=CONFIG['base_seed'])
    p.add_argument('--lr',type=float)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(4)
    run_case(a.source,a.split,a.schema,a.seed,a.mode,a.smoke,a.lr)
