import config
import numpy as np
import tensorflow as tf

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
        jaccard_coef = 1.
    else:
        jaccard_coef = tp / (tp + fp + fn)
        
    return jaccard_coef

def dice(a, b):
    tp, fp, fn = get_tp_fp_fn(a, b)
    
    dice_coef = None
    if tp + fp + fn == 0:
        dice_coef = 1.
    else:
        dice_coef = (2 * tp) / (2 * tp + fp + fn)
    
    return dice_coef

def jaccard_and_dice(preds, gts, thr):
    jaccard_and_dice_res = np.zeros((len(preds), 2))
    
    for idx, (pred, gt) in enumerate(zip(preds, gts)):
        pred = (pred >= thr)
        
        jaccard_coef = jaccard(pred, gt)
        dice_coef = dice(pred, gt)
        
        jaccard_and_dice_res[idx] = (jaccard_coef, dice_coef)
        
    return jaccard_and_dice_res

# tensorflow implementation (with thr)
def tf_get_tp_fp_fn(a_in, b_in):
    a = tf.greater_equal(a_in, config.PRED_THR)
    not_a = tf.logical_not(a)
    b = tf.greater_equal(b_in, config.PRED_THR)
    not_b = tf.logical_not(b)
    
    tp_and = tf.logical_and(a, b)
    tp_count = tf.count_nonzero(tp_and)
    tp = tf.cast(tp_count, tf.float64)
    
    fp_and = tf.logical_and(a, not_b)
    fp_count = tf.count_nonzero(fp_and)
    fp = tf.cast(fp_count, tf.float64)
    
    fn_and = tf.logical_and(not_a, b)
    fn_count = tf.count_nonzero(fn_and)
    fn = tf.cast(fn_count, tf.float64)
    
    return tp, fp, fn

def tf_jaccard(a, b):
    tp, fp, fn = tf_get_tp_fp_fn(a, b)
    jaccard_coef = tf.cond(tf.equal(tp + fp + fn, 0),
                           lambda: tf.constant(1, tf.float64),
                           lambda: tp / (tp + fp + fn))
    return jaccard_coef

def tf_dice(a, b):
    tp, fp, fn = tf_get_tp_fp_fn(a, b)
    dice_coef = tf.cond(tf.equal(tp + fp + fn, 0),
                        lambda: tf.constant(1, tf.float64),
                        lambda: (2 * tp) / (2 * tp + fp + fn))
    return dice_coef