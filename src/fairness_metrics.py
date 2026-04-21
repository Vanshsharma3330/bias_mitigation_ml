import numpy as np

def demographic_parity(y_pred, protected):
    g0 = y_pred[protected == 0]
    g1 = y_pred[protected == 1]
    return abs(np.mean(g0) - np.mean(g1))

def equal_opportunity(y_true, y_pred, protected):
    mask0 = (protected == 0) & (y_true == 1)
    mask1 = (protected == 1) & (y_true == 1)

    tpr0 = np.mean(y_pred[mask0]) if mask0.sum() else 0
    tpr1 = np.mean(y_pred[mask1]) if mask1.sum() else 0

    return abs(tpr0 - tpr1)