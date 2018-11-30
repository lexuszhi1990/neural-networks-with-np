import numpy as np
from sklearn.metrics import roc_curve

y = np.array([0,0,1,1])
pred = np.array([0.1,0.4,0.35,0.8])

data = np.array([
    [1, .9],
    [1, .8],
    [0, .7],
    [1, .6],
    [1, .55],
    [1, .54],
    [0, .53],
    [0, .52],
    [1, .51],
    [0, .505],
    [1, .4],
    [0, .39],
    [1, .38],
    [0, .37],
    [0, .36],
    [0, .35],
    [1, .34],
    [0, .33],
    [1, .30],
    [0, .1]
])
y = data[:, 0]
pred = data[:, 1]


fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
print(fpr)
print(tpr)
print(thresholds)

from sklearn.metrics import auc
print(auc(fpr, tpr))


import pdb; pdb.set_trace()


def roc(y, pred):
    thresholds = sorted(np.unique(pred).tolist())
    thresholds.append(pred.max() + 1)

    tpr = []
    fpr = []
    ths = []
    for th in thresholds:

        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(len(pred)):

            if pred[i] >= th:
                if y[i] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if y[i] == 0:
                    tn += 1
                else:
                    fn += 1
        fpr.append(fp/(fp+tn))
        tpr.append(tp/(tp+fn))
        ths.append(th)

    return fpr, tpr, ths


fpr_, tpr_, thresholds_ = roc(y, pred)
print(fpr_)
print(tpr_)
print(thresholds_)

def auc_area(fpr, tpr):
    area = 0.
    for i in range(len(fpr) -1):
        area += (fpr[i+1] - fpr[i]) * tpr[i]

    return area

print(auc_area(fpr, tpr))

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(fpr, tpr)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic')
# plt.show()

import pdb; pdb.set_trace()
