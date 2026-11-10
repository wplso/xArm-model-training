"""Small CPU autoencoders, masked reconstruction and training-only preprocessing."""
import copy,random,time
import numpy as np,torch
from torch import nn
class Normalizer:
    def fit(self,X):
        X=np.asarray(X,float);self.sequence=X.ndim==3
        if self.sequence:
            a=X.transpose(0,2,1).reshape(-1,X.shape[1]);self.fill=np.nanmean(a,axis=0)
        else:a=X;self.fill=np.nanmedian(a,axis=0)
        if not np.isfinite(self.fill).all():raise ValueError('All-missing training channel')
        a=np.where(np.isfinite(a),a,self.fill);self.center=a.mean(axis=0);self.scale=a.std(axis=0)
        self.scale[self.scale<1e-8]=1.
        return self
    def transform(self,X):
        X=np.asarray(X,float);mask=np.isfinite(X)
        shape=(1,-1,1) if self.sequence else (1,-1)
        fill=self.fill.reshape(shape);center=self.center.reshape(shape);scale=self.scale.reshape(shape)
        Z=(np.where(mask,X,fill)-center)/scale
        return Z.astype(np.float32),mask.astype(np.float32)
    def save(self,path):np.savez(path,fill=self.fill,center=self.center,scale=self.scale,sequence=self.sequence)

class MLP(nn.Module):
    def __init__(self,shape,hidden,latent):
        super().__init__();self.shape=tuple(shape);n=int(np.prod(shape))
        self.net=nn.Sequential(nn.Linear(n,hidden),nn.ReLU(),nn.Linear(hidden,latent),nn.ReLU(),
            nn.Linear(latent,hidden),nn.ReLU(),nn.Linear(hidden,n))
    def forward(self,x):return self.net(x.flatten(1)).reshape(x.shape[0],*self.shape)

class CNN(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.net=nn.Sequential(nn.Conv1d(channels,8,5,2,2),nn.ReLU(),nn.Conv1d(8,16,5,2,2),nn.ReLU(),
            nn.Conv1d(16,4,3,2,1),nn.ReLU(),nn.ConvTranspose1d(4,16,4,2,1),nn.ReLU(),
            nn.ConvTranspose1d(16,8,4,2,1),nn.ReLU(),nn.ConvTranspose1d(8,channels,4,2,1))
    def forward(self,x):return self.net(x)

def make_model(architecture,shape,cfg):
    if architecture=='feature_MLP_AE':return MLP(shape,cfg['feature_MLP_hidden'],cfg['feature_MLP_latent'])
    if architecture=='sequence_MLP_AE':return MLP(shape,cfg['sequence_MLP_hidden'],cfg['sequence_MLP_latent'])
    if architecture=='sequence_CNN1D_AE':return CNN(shape[0])
    raise ValueError(architecture)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)

def masked_scores(pred,x,mask):
    dims=tuple(range(1,x.ndim));count=mask.sum(dim=dims)
    if torch.any(count==0):raise ValueError('No valid inputs in cycle')
    return (((pred-x)**2)*mask).sum(dim=dims)/count

def score(model,X,mask,batch_size=64):
    model.eval();values=[]
    with torch.inference_mode():
        for begin in range(0,len(X),batch_size):
            x=torch.from_numpy(X[begin:begin+batch_size]);m=torch.from_numpy(mask[begin:begin+batch_size])
            values.append(masked_scores(model(x),x,m).numpy())
    return np.concatenate(values)

def train_model(architecture,shape,X,mask,cfg,seed,stop=None,epochs=None):
    seed_all(seed);model=make_model(architecture,shape,cfg)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    generator=torch.Generator().manual_seed(seed);history=[];best=float('inf');best_epoch=0;stale=0
    xt=torch.from_numpy(X);mt=torch.from_numpy(mask)
    maximum=epochs if epochs is not None else cfg['max_epochs']
    for epoch in range(1,maximum+1):
        model.train();order=torch.randperm(len(X),generator=generator);total=0.
        for idx in order.split(cfg['batch_size']):
            optimizer.zero_grad(set_to_none=True);loss=masked_scores(model(xt[idx]),xt[idx],mt[idx]).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite training loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(idx)
        row=dict(epoch=epoch,train_loss=total/len(X))
        if stop is not None:
            current=float(score(model,*stop).mean());row['inner_stop_loss']=current
            if current<best-cfg['min_delta']:best=current;best_epoch=epoch;stale=0
            else:stale+=1
            history.append(row)
            if stale>=cfg['patience']:break
        else:history.append(row)
    return model,history,(best_epoch if stop is not None else maximum)

def threshold(scores):return float(np.quantile(scores,.95,method='higher'))
