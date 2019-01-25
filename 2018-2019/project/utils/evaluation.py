import numpy as np

def get_tp_fp_fn(a, b):
    a = np.equal(a, 1)
    not_a = np.logical_not(a)
    b = np.equal(b, 1)
    not_b = np.logical_not(b)
    
    tp = np.logical_and(a, b).sum().astype(np.float64)
    fp = np.logical_and(a, not_b).sum().astype(np.float64)
    fn = np.logical_and(not_a, b).sum().astype(np.float64)
    
    return tp, fp, fn

def jaccard(a, b):
    tp, fp, fn = get_tp_fp_fn(a, b)
    
    jaccard_coef = None
    if tp + fp + fn == 0:
        jaccard_coef = 1
    else:
        jaccard_coef = tp / (tp + fp + fn)
        
    return jaccard_coef

def dice(a, b):
    tp, fp, fn = get_tp_fp_fn(a, b)
    
    dice_coef = None
    if tp + fp + fn == 0:
        dice_coef = 1
    else:
        dice_coef = (2 * tp) / (2 * tp + fp + fn)
    
    return dice_coef