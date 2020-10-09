import pandas as pd
import numpy as np
import scipy.stats
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import RepeatedStratifiedKFold
import tensorflow as tf
import matplotlib.pyplot as plt

#############################################################################
# user defined funs

'''
function details are follwings:

threshold_decision: According to thresholds we are interested,
calculate precision_ppvs, sensitivity_recalls, specificities, npvs, accs, f1s in the specific model 

cut_off_decision: According to thresholds we are interested,
calculate precision_ppvs, sensitivity_recalls, specificities, npvs, accs, f1s
in the numerous models from 3 repeated 5 fold cv

mean_performances_to_df: Calulate average of each performance achieved from 3 repeated 5 fold cv,
and then make them to dataframe

mean_confidence_interval: Calculate mean performance with 95% CI

performances_hard_decision: caculate performances (PPV, NPV, sensitivity, specificity, accuracy, f1)
at the selected threshold in the model
'''

def threshold_decision(threshold_of_interests, probas_, y_test1, decimals=5):
    
    precision_ppvs = []
    sensitivity_recalls = []
    specificities = []
    npvs = []
    accs = []
    f1s = []

    for threshold_of_interest in threshold_of_interests:

        y_pred = probas_ >= threshold_of_interest

        tn, fp, fn, tp = confusion_matrix(y_test1, y_pred).ravel()

        precision_ppv = tp / (tp+fp); precision_ppvs.append(np.around(precision_ppv, decimals=decimals))
        sensitivity_recall = tp / (tp+fn); sensitivity_recalls.append(np.around(sensitivity_recall, decimals=decimals))
        specificity = tn / (tn+fp); specificities.append(np.around(specificity, decimals=decimals))
        npv = tn / (tn+fn); npvs.append(np.around(npv, decimals=decimals))
        acc = (tp+tn) / (tp+tn+fp+fn); accs.append(np.around(acc, decimals=decimals))
        f1 = (2 * precision_ppv * sensitivity_recall) / (precision_ppv + sensitivity_recall); f1s.append(np.around(f1, decimals=decimals))

    return(precision_ppvs, sensitivity_recalls, specificities, npvs, accs, f1s)

def cut_off_decision(clf, X_trainval, threshold_of_interests):
    
    precision_ppvs_all = []
    sensitivity_recalls_all = []
    specificities_all = []
    npvs_all = []
    accs_all = []
    f1s_all = []
    aucs = []

    rskfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)

    for train_index, test_index in rskfold.split(X_trainval, y_trainval):

        X_train1, X_test1 = X_trainval.iloc[train_index], X_trainval.iloc[test_index]
        y_train1, y_test1 = y_trainval[train_index], y_trainval[test_index]

        clf_rf = clf
        clf_rf.fit(X_train1, y_train1)
        probas_rf = clf_rf.predict_proba(X_test1)

        fpr, tpr, thresholds = roc_curve(y_test1, probas_rf[:, 1])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        precision_ppvs, sensitivity_recalls, specificities, npvs, accs, f1s = threshold_decision(threshold_of_interests, probas_rf[:, 1], y_test1)

        precision_ppvs_all.append(precision_ppvs)
        sensitivity_recalls_all.append(sensitivity_recalls)
        specificities_all.append(specificities)
        npvs_all.append(npvs)
        accs_all.append(accs)
        f1s_all.append(f1s)
    
    return(precision_ppvs_all, sensitivity_recalls_all, specificities_all, npvs_all, accs_all, f1s_all, aucs)

def mean_performances_to_df(precision_ppvs, sensitivity_recalls, specificities, npvs, accs, f1s):
    
    mean_precision_ppvs = np.mean(precision_ppvs, axis=0)
    mean_sensitivity_recalls = np.mean(sensitivity_recalls, axis=0)
    mean_specificities = np.mean(specificities, axis=0)
    mean_npvs = np.mean(npvs, axis=0)
    mean_accs = np.mean(accs, axis=0)
    mean_f1s = np.mean(f1s, axis=0)
    
    data= {'Threshold': threshold_of_interests,
       'Precision(PPV)': mean_precision_ppvs,
       'Sensitivity(Recall)': mean_sensitivity_recalls,
       'Specificity': mean_specificities,
       'NPV': mean_npvs,
       'F1': mean_f1s,
       'Accuracy': mean_accs}

    df = pd.DataFrame(data)
    
    return df

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def performances_hard_decision(y_test, y_proba, threshold_of_interest):
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    print("AUC: {:.3f}".format(roc_auc))
    
    y_pred = y_proba >= threshold_of_interest
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    ppv = tp / (tp+fp)
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    npv = tn / (tn+fn)
    accuracy = (tp+tn) / (tp+tn+fp+fn)
    f1 = (2 * ppv * sensitivity) / (ppv + sensitivity)
    
    print("specificity: {:.3f}".format(specificity))
    print("sensitivity: {:.3f}".format(sensitivity))
    print("PPV: {:.3f}".format(ppv))
    print("NPV: {:.3f}".format(npv))
    print("f1: {:.3f}".format(f1))
    print("accuracy: {:.3f}".format(accuracy))
    print("threshold: {:.3f}".format(threshold_of_interest))
    
    
#############################################################################
# random forest cut off decision (sensitivity >= 0.85)
threshold_of_interests = [i for i in np.arange(0, 1, 0.001)]
precision_ppvs, sensitivity_recalls, specificities, npvs, accs, f1s, aucs = cut_off_decision(clf_rf, X_trainval_selected, threshold_of_interests)

rf_data = mean_performances_to_df(precision_ppvs, sensitivity_recalls, specificities, npvs, accs, f1s)

rf_selected_thres = rf_data[rf_data['Sensitivity(Recall)'] >= 0.85].sort_values(['Sensitivity(Recall)', 'Specificity'] , ascending=True, ignore_index=True).loc[0, 'Threshold']

# internal validity with AUROCs from the 3 repeaped 5 fold cv
np.around(mean_confidence_interval(aucs, confidence=0.95), decimals=3)

# final performances (PPV, NPV, sensitivity, specificity, accuracy, f1) at the selected threshold in the model
probas_rf = clf_rf.predict_proba(X_test_selected)
performances_hard_decision(y_test, probas_rf[: ,1], rf_selected_thres)


#############################################################################
# svm cut off decision (sensitivity >= 0.85)
threshold_of_interests = [i for i in np.arange(0, 1, 0.001)]
precision_ppvs, sensitivity_recalls, specificities, npvs, accs, f1s, aucs = cut_off_decision(clf_svm, X_trainval_selected, threshold_of_interests)

svm_data = mean_performances_to_df(precision_ppvs, sensitivity_recalls, specificities, npvs, accs, f1s)

svm_selected_thres = svm_data[svm_data['Sensitivity(Recall)'] >= 0.85].sort_values(['Sensitivity(Recall)', 'Specificity'] , ascending=True, ignore_index=True).loc[0, 'Threshold']

# internal validity with AUROCs from the 3 repeaped 5 fold cv
np.around(mean_confidence_interval(aucs, confidence=0.95), decimals=3)

# final performances (PPV, NPV, sensitivity, specificity, accuracy, f1) at the selected threshold in the model
probas_svm = clf_svm.predict_proba(X_test_selected)
performances_hard_decision(y_test, probas_svm[: ,1], svm_selected_thres)


#############################################################################
# lr cut off decision (sensitivity >= 0.85)
threshold_of_interests = [i for i in np.arange(0, 1, 0.001)]
precision_ppvs, sensitivity_recalls, specificities, npvs, accs, f1s, aucs = cut_off_decision(clf_lr, X_trainval_selected, threshold_of_interests)

lr_data = mean_performances_to_df(precision_ppvs, sensitivity_recalls, specificities, npvs, accs, f1s)

lr_selected_thres = lr_data[lr_data['Sensitivity(Recall)'] >= 0.85].sort_values(['Sensitivity(Recall)', 'Specificity'] , ascending=True, ignore_index=True).loc[0, 'Threshold']

# internal validity with AUROCs from the 3 repeaped 5 fold cv
np.around(mean_confidence_interval(aucs, confidence=0.95), decimals=3)

# final performances (PPV, NPV, sensitivity, specificity, accuracy, f1) at the selected threshold in the model
probas_lr = clf_lr.predict_proba(X_test_selected)
performances_hard_decision(y_test, probas_lr[: ,1], lr_selected_thres)


#############################################################################
# ann cut off decision (sensitivity >= 0.85)
input_col_num = X_test_selected.shape[1]

## temporarily used function
def find_best_epoch_in(model):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 100)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose = 1,
                                  patience=5, min_lr=1e-15)

    callbacks = [reduce_lr,early_stopping]

    try_epoch = 1000

    history = model.fit(X_train2,
                      y_train2,
                      epochs=try_epoch,
                      batch_size=128,
                      validation_data=(X_valid, y_valid),
                      callbacks=callbacks)
    return (np.argmin(history.history['val_loss']) + 1)

threshold_of_interests = [i for i in np.arange(0, 1, 0.001)]

i = 0
ann_aucs = []
precision_ppvs_all = []
sensitivity_recalls_all = []
specificities_all = []
npvs_all = []
accs_all = []
f1s_all = []

rskfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)

for train_index, test_index in rskfold.split(X_trainval_selected, y_trainval):
    
    i += 1
    
    X_train1, X_test1 = X_trainval_selected.iloc[train_index], X_trainval_selected.iloc[test_index]
    y_train1, y_test1 = y_trainval[train_index], y_trainval[test_index]
    
    #----- split into train and validation set
    X_train2, X_valid, y_train2, y_valid = train_test_split(X_train1, y_train1,
                                                            test_size = 0.2, stratify=y_train1, random_state=0)
    
    #------ find the best epoch
    model = make_model(input_col_num)
    best_epoch_in = find_best_epoch_in(model)
    
    #----- model building with trainval set
    model = make_model(input_col_num)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 100)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, verbose = 1,
                                  patience=5, min_lr=1e-15)

    save_best = tf.keras.callbacks.ModelCheckpoint('./cv_best_models/best_model_' + str(i) + '.h5', monitor = 'loss', verbose = 1, save_bset_only = True, save_weights_only=False, mode = 'min')
    callbacks = [reduce_lr, save_best]

    model.fit(X_train1, y_train1, epochs=best_epoch_in, batch_size=128, callbacks = callbacks)
    
    #----- evaluate the model
    clf_ann = tf.keras.models.load_model('./cv_best_models/best_model_' + str(i) + '.h5')
    probas_ = clf_ann.predict(X_test1)

    #----- Compute ROC curve and area under the curve
    fpr, tpr, thresholds = roc_curve(y_test1, probas_) 
    roc_auc = auc(fpr, tpr)
    ann_aucs.append(roc_auc)
    
    #----- cut offs
    precision_ppvs, sensitivity_recalls, specificities, npvs, accs, f1s = threshold_decision(threshold_of_interests, probas_, y_test1)
    precision_ppvs_all.append(precision_ppvs)
    sensitivity_recalls_all.append(sensitivity_recalls)
    specificities_all.append(specificities)
    npvs_all.append(npvs)
    accs_all.append(accs)
    f1s_all.append(f1s)
    
mean_precision_ppvs = np.mean(precision_ppvs_all, axis=0)
mean_sensitivity_recalls = np.mean(sensitivity_recalls_all, axis=0)
mean_specificities = np.mean(specificities_all, axis=0)
mean_npvs = np.mean(npvs_all, axis=0)
mean_accs = np.mean(accs_all, axis=0)
mean_f1s = np.mean(f1s_all, axis=0)
    
data= {'Threshold': threshold_of_interests,
       'Precision(PPV)': mean_precision_ppvs,
       'Sensitivity(Recall)': mean_sensitivity_recalls,
       'Specificity': mean_specificities,
       'NPV': mean_npvs,
       'F1': mean_f1s,
       'Accuracy': mean_accs}

df = pd.DataFrame(data)

ann_selected_thres = df[df['Sensitivity(Recall)'] >= 0.85].sort_values(['Sensitivity(Recall)', 'Specificity'] , ascending=True, ignore_index=True).loc[0, 'Threshold']

# internal validity with AUROCs from the 3 repeaped 5 fold cv
np.around(mean_confidence_interval(ann_aucs, confidence=0.95), decimals=3)

# final performances (PPV, NPV, sensitivity, specificity, accuracy, f1) at the selected threshold in the model
y_pred = ANN_model.predict(X_test_selected)
performances_hard_decision(y_test, y_pred, ann_selected_thres)


#############################################################################
# ROC curves for all models
plt.rcParams["figure.figsize"] = (8,8)
plt.rcParams.update({'axes.labelsize': 'large'})

## RF, ANN, SVM. LR
y_pred1 = clf_rf.predict_proba(X_test_selected)[:, 1]
y_pred2 = ANN_model.predict(X_test_selected)
y_pred3 = clf_svm.predict_proba(X_test_selected)[:, 1]
y_pred4 = clf_lr.predict_proba(X_test_selected)[:, 1]

fpr1, tpr1, thresholds1 = roc_curve(y_test, y_pred1)
fpr2, tpr2, thresholds2 = roc_curve(y_test, y_pred2)
fpr3, tpr3, thresholds3 = roc_curve(y_test, y_pred3)
fpr4, tpr4, thresholds4 = roc_curve(y_test, y_pred4)

roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)
roc_auc4 = auc(fpr4, tpr4)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#7f7f7f', label='Chance', alpha=.8)

plt.plot(fpr1, tpr1, color='#1f77b4',
         label=r'RF (AUROC = %0.3f)' % (roc_auc1),
         lw=2, alpha=.8)

plt.plot(fpr4, tpr4, color='#ff7f0e',
         label=r'LR (AUROC = %0.3f)' % (roc_auc4),
         lw=2, alpha=.8)

plt.plot(fpr2, tpr2, color='#2ca02c',
         label=r'ANN (AUROC = %0.3f)' % (roc_auc2),
         lw=2, alpha=.8)

plt.plot(fpr3, tpr3, color='#d62728',
         label="SVM (AUROC = %0.3f)" %(roc_auc3),
         lw=2, alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC(Receiver operating characteristic) Curve\n')
plt.legend(loc="lower right")
plt.savefig('UE_ROC_curve_all', dpi=500)
plt.show()
