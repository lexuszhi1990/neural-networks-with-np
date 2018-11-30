from sklearn import metrics
import numpy as np

### precision

y_true = np.array([ 1, 10, 35,  9,  7, 29, 26,  3,  8, 23, 39, 11, 20,  2,  5, 23, 28,
       30, 32, 18,  5, 34,  4, 25, 12, 24, 13, 21, 38, 19, 33, 33, 16, 20,
       18, 27, 39, 20, 37, 17, 31, 29, 36,  7,  6, 24, 37, 22, 30,  0, 22,
       11, 35, 30, 31, 14, 32, 21, 34, 38,  5, 11, 10,  6,  1, 14, 12, 36,
       25,  8, 30,  3, 12,  7,  4, 10, 15, 12, 34, 25, 26, 29, 14, 37, 23,
       12, 19, 19,  3,  2, 31, 30, 11,  2, 24, 19, 27, 22, 13,  6, 18, 20,
        6, 34, 33,  2, 37, 17, 30, 24,  2, 36,  9, 36, 19, 33, 35,  0,  4,
        1])
y_pred = np.array([ 1, 10, 35,  7,  7, 29, 26,  3,  8, 23, 39, 11, 20,  4,  5, 23, 28,
       30, 32, 18,  5, 39,  4, 25,  0, 24, 13, 21, 38, 19, 33, 33, 16, 20,
       18, 27, 39, 20, 37, 17, 2, 29, 36,  7,  6, 24, 37, 22, 30,  0, 22,
       11, 35, 30, 31, 14, 32, 21, 34, 38,  5, 11, 10,  6,  1, 14, 30, 36,
       25,  8, 30,  3, 12,  7,  4, 10, 15, 12,  4, 22, 26, 29, 14, 37, 23,
       12, 19, 2,  3, 25, 31, 30, 11, 25, 24, 19, 27, 22, 13,  6, 18, 20,
        6, 39, 33,  9, 37, 17, 30, 24,  9, 36, 39, 36, 19, 33, 35,  0,  4,
        1])

# print(metrics.precision_recall_fscore_support(y_true, y_pred, average='macro'))
# print(metrics.precision_recall_fscore_support(y_true, y_pred, average='micro'))
# print(metrics.precision_recall_fscore_support(y_true, y_pred, average='weighted'))
# print(metrics.precision_recall_fscore_support(y_true, y_pred, average=None))

def precision_macro(y_true, y_pred, eps=1e-20):
    cls_ids = set(y_pred.tolist() + y_true.tolist())
    details = []
    for cls_id in cls_ids:

        tp = (y_true[y_pred == cls_id] == cls_id).sum()
        fp = (y_true[y_pred == cls_id] != cls_id).sum()
        assert tp+fp == len(y_true[y_pred == cls_id])
        fn = (y_pred[y_true == cls_id] != cls_id).sum()

        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = 2 * p * r / (p + r + eps)
        details.append([cls_id, p, r, f1])
    details = np.array(details)

    return details[:, 1].mean(), details[:, 2].mean(), details[:, 3].mean()


def precision_micro(y_true, y_pred, eps=1e-20):
    truepos = y_pred == y_true
    p = truepos.sum() / (len(y_pred) + eps)
    r = truepos.sum() / (len(y_true) + eps)
    f1 = 2 * p * r / ( p + r + eps)

    return p, r, f1


print("Precision micro (sklearn): ", metrics.precision_recall_fscore_support(y_true, y_pred, average='micro')[:3])
print("Precision micro     (own): ", precision_micro(y_true, y_pred)[:3])

print("Precision macro (sklearn): ", metrics.precision_recall_fscore_support(y_true, y_pred, average='macro')[:3])
print("Precision macro     (own): ", precision_macro(y_true, y_pred)[:3])


import pdb; pdb.set_trace()

### multi labels

y_pred = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
])

y_true = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
])

def f1_micro(y_true, y_preds, thresh=0.5, eps=1e-20):
    # binary representation from probabilities (not relevant)
    preds_bin = y_preds > thresh
    truepos = preds_bin * y_true

    # take sums and calculate precision on scalars
    p = truepos.sum() / (preds_bin.sum() + eps)
    # take sums and calculate recall on scalars
    r = truepos.sum() / (y_true.sum() + eps)

    f1 = 2*p*r / (p+r+eps) # we calculate f1 on scalars
    return f1

def f1_macro(y_true, y_preds, thresh=0.5, eps=1e-20):
    # binary representation from probabilities (not relevant)
    preds_bin = y_preds > thresh
    truepos = preds_bin * y_true

    p = truepos.sum(axis=0) / (preds_bin.sum(axis=0) + eps)
    # sum along axis=0 (classes)
    # and calculate precision array
    r = truepos.sum(axis=0) / (y_true.sum(axis=0) + eps)
    # sum along axis=0 (classes)
    #  and calculate recall array

    f1 = 2*p*r / (p+r+eps) # we calculate f1 on arrays
    return np.mean(f1) # we take the average of the individual f1 scores at the very end!


print('Micro F1 (sklearn):',metrics.f1_score(y_true, y_pred, average='micro'))
print('Macro F1 (sklearn):',metrics.f1_score(y_true, y_pred, average='macro'))
print('Micro F1 (own)    :',f1_micro(y_true, y_pred))
print('Macro F1 (own)    :',f1_macro(y_true, y_pred))
