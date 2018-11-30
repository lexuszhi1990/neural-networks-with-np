import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


if __name__ == '__main__':

    #file_dir = '/apps/model/resnext/50/679class/pretrain4719/without-overfit/res/'
    file_dir = '/apps/model/deploy/test/'
    with open(file_dir+'newprob_res_9.txt','r') as of:
        lines = of.readlines()
    labels = []
    pred_probs_204 = []
    pred_labels_204 = []
    pred_probs = []
    pred_labels = []
    for line in lines:
        item  = line.strip().split('\t')
        labels.append(int(item[0]))
        pred_probs_204.append(float(item[1]))
        pred_labels_204.append(int(item[2]))
        pred_probs.append(float(item[3]))
        pred_labels.append(int(item[4]))
    xs = []
    ys = []
    thresvalue =[]
    num_labeled = 67514.0
    num_other = float(len(labels)-num_labeled)
    wf = open('/home/zy/xabs-0.05-1.txt','w')
    #threshold1 = [x for x in np.arange(0, 0.99, 0.1)]
    threshold2 = [x for x in np.arange(0.97, 0.98, 0.0001)]
    thresholdI = threshold2
    threshold3 = [x for x in np.arange(0.57, 0.70 ,0.001)]
    #threshold4 = [x for x in np.arange(0.15, 1, 0.1)]
    thresholdJ = threshold3
    print(len(thresholdI))
    for xx,thresj in enumerate(thresholdJ):
        print(xx,' ',thresj)
        for thresi in thresholdI:
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            nocheck_labeled_b = 0
            nocheck_labeled_s = 0
            nocheck_other_b = 0
            nocheck_other_s = 0
            human_tp = 0
            human_fp = 0
            human_tn = 0
            human_fn = 0
            check = 0
            for i in range(len(labels)):
                if labels[i] != -1:
                    if pred_probs[i]>thresi:
                        if pred_labels[i] == labels[i]:
                            tp += 1
                        else:
                            fp += 1
                        nocheck_labeled_b += 1
                    else:
                        if pred_probs_204[i]<thresj:
                            fn += 1
                            nocheck_labeled_s += 1
                        else:
                            if pred_labels[i] == labels[i]:
                                human_tp += 1
                            else:
                                human_fp += 1
                            check += 1

                else:
                    if pred_probs[i]>thresi:
                        if pred_labels[i] > 203:
                            tp += 1
                        else:
                            fp += 1
                        nocheck_other_b += 1
                    else:
                        if  pred_probs_204[i]<thresj:
                            tn += 1
                            nocheck_other_s += 1
                        else:
                            if pred_labels[i] > 203:
                                human_tn += 1
                            else:
                                human_fn += 1
                            check += 1

            nocheck = nocheck_labeled_b + nocheck_labeled_s + nocheck_other_b + nocheck_other_s
            if tp + fp + tn + fn ==0:
                x=0.0
                y=0.0
            else:
                x = float(fp + fn)/float(tp + fp + tn + fn)
                y = float(nocheck)/float(len(labels))
            xs.append(x)
            ys.append(y)
            check_acc_l = float(human_tp) / float(human_tp + human_fp+human_fn+human_tn)  # +human_fn+human_tn)
            check_acc = float(human_tn)/float(human_tp + human_fp+human_fn+human_tn)#+human_fn+human_tn)
            total_acc = float(human_tp+tp+tn+human_tn)/float(len(labels))
            if x<0.0501 and x>0.0499:
                print(thresi,' ',thresj, ' ', x, ' ', y,' ',nocheck_labeled_b+nocheck_other_b,' ',nocheck_labeled_s+nocheck_other_s ,' ',check_acc_l,' ',human_tp + human_fp,'',check_acc,' ',human_fn+human_tn,' ',human_fp+human_fn)
                wf.write(str(thresi) + '\t' +str(float(fp)/float(tp+fp+tn+fn))+'\t'+ str(nocheck_labeled_b + nocheck_other_b) + '\t' + str(thresj) + '\t'+str(float(fn)/float(tn+fn+tp+fp))+'\t' + str(
                    nocheck_labeled_s + nocheck_other_s) + '\t' + str(x) + '\t'+str(y)+'\t'+str(check_acc_l)+'\t'+str(human_tp + human_fp)+'\t'+str(check_acc)+'\t'+str(human_fn+human_tn)+'\t'+str(human_fp+human_fn)+'\t'+
                         str(float(human_tp + human_tn)/float(human_fp+human_fn+human_tp + human_tn))+'\t'+str(total_acc)+'\n')

            #print(dropout_labeled+dropout_other+check+nocheck)
            thresvalue.append(thresj)
            #wf.write(str(x)+'\t'+str(y)+'\t'+str(thresi)+'\t'+str(thresj)+'\t'+str(nocheck)+'\t'+str(fpless)+'\t'+str(tpless)+'\n')
"""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    plt.ylim(ymax=1, ymin=0)
    plt.grid()
    plt.xlabel('fpr')
    plt.ylabel('auto/thres/miss')
    plt.title('auto-thres-679class_4719class')
    ax.plot(xs, ys, color='red')
    ax.plot(xs, thresvalue, color='green')
    plt.legend(('auto', 'thres'), loc='best')
    plt.show()
"""





