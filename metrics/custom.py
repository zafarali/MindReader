import numpy as np
from sklearn.metrics import roc_auc_score

def multiple_auc(Y_actual, Y_pred, return_individual=False):
    """
        Calculates the averaged ROC for each class
        @params:
            Y_actual: true values of the labels shape:(n , 1)
            Y_pred: predicted values of the labels shape:(n , 1)
        @returns:
            [0] roc_auc for each class (dict)
            [1] averaged_roc amongst classes (float)
    """
    uniques = np.unique(Y_actual)
#     print uniques
    roc_aucs = {}
    for label in uniques:
#         print label
        Y_a = Y_actual.copy()
        Y_p = Y_pred.copy()
        
        matches = Y_a == label
        Y_a[matches] = -1
        Y_a[np.invert(matches)] = 0
        Y_a[Y_a == -1] = 1
        
        matches = Y_p == label
        Y_p[matches] = -1
        Y_p[np.invert(matches)] = 0
        Y_p[Y_p == -1] = 1
        roc_aucs[label] = roc_auc_score(Y_a, Y_p)
    averaged_roc = np.mean(roc_aucs.values())
    if return_individual:
        return roc_aucs, averaged_roc
    else:
        return averaged_roc
    