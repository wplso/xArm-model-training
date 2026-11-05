"""Original native-sample statistics and current-fall definitions."""
import math
import statistics as st
import numpy as np

def describe(values,prefix):
    x=[float(v) for v in values if math.isfinite(float(v))]
    if not x:return {prefix+'_n':0}
    return {prefix+'_n':len(x),prefix+'_mean':st.fmean(x),prefix+'_rms':math.sqrt(st.fmean(a*a for a in x)),
        prefix+'_std':st.pstdev(x),prefix+'_min':min(x),prefix+'_max':max(x),prefix+'_range':max(x)-min(x),
        prefix+'_abs_peak':max(map(abs,x)),prefix+'_abs_p95_nearest_rank':sorted(map(abs,x))[math.ceil(.95*len(x))-1]}

def window(t,start,end):return (np.asarray(t)>=start)&(np.asarray(t)<end)

def current_fall(t,current):
    high=np.median(current[window(t,4.5,5.3)]);low=np.median(current[window(t,6.5,6.9)])
    threshold=(high+low)/2
    for i in range(len(t)-1):
        if 5.2<=t[i]<6.6 and current[i]<threshold and current[i+1]<threshold:
            return float(t[i]),float(threshold)
    return np.nan,float(threshold)
