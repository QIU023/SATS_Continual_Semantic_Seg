import os
# import torch
import numpy as np
from tqdm import tqdm
print(os.getcwd())
fdir = '/home/jovyan/2D_CIL_Seg/SATS/ensemble_output'

from metrics.stream_metrics import StreamSegMetrics

metric = StreamSegMetrics(21)

f_arr = []

# for fpath in os.listdir(fdir):
fpath = os.path.join(fdir, 'segformer_b2')
fpath2 = os.path.join(fdir, 'deeplab')

deeplab_list = os.listdir(fpath2)

for fitem in tqdm(os.listdir(fpath)):
    if 'logit' in fitem:
        fname = os.path.join(fpath, fitem)
        fname2 = os.path.join(fpath2, fitem)
    
        arr = fitem.split('_')[-1].split('.')
        arr[0] = 'gt'
        arr = '.'.join(arr)
        arr = fitem.split('_')[:-1]+[arr]
        gtitem = '_'.join(arr)
        
        
        gtname = os.path.join(fpath, gtitem)
        gtname2 = os.path.join(fpath2, gtitem)

        segformer_logit = np.load(fname)
        segformer_gt = np.load(gtname)

        deeplab_logit = np.load(fname2)
        deeplab_gt = np.load(gtname)

        assert (segformer_gt == deeplab_gt).all()
        ensemble_logit = (segformer_logit + deeplab_logit) / 2
        
        metric.update(deeplab_gt, ensemble_logit.argmax(axis=1))
    
print(metric.to_str(metric.get_results()))
