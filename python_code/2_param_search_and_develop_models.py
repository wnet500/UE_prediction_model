import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import tensorflow as tf


#############################################################################
# Random Foest - grid search
n_estimators = [100, 200, 300, 500, 1000, 1500, 2000, 2500]
param_grid = {'n_estimators': n_estimators}
skfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=0), param_grid, cv=skfold,
                           scoring='roc_auc', iid=False,
                           n_jobs=-1, return_train_score=True)
grid_search_rf.fit(X_trainval_selected, y_trainval)

print('Best params: {}'.format(grid_search_rf.best_params_))

clf_rf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1) # rf model with the best params
clf_rf.fit(X_trainval_selected, y_trainval)


#############################################################################
# SVM - grid search
C = np.logspace(-6, 3, 10)
gamma = np.logspace(-6, 3, 10)
param_grid = {'C': C, 'gamma': gamma}
skfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)

grid_search_svm = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, cv=skfold,
                           scoring='roc_auc', iid=False,
                           n_jobs=-1, return_train_score=True)
grid_search_svm.fit(X_trainval_selected, y_trainval)

print('Best params: {}'.format(grid_search_svm.best_params_))

clf_svm = SVC(kernel='rbf', C=0.1, gamma=0.1, probability=True) # svm model with the best params
clf_svm.fit(X_trainval_scaled_selected, y_trainval)


#############################################################################
# Logistic Resgrssion - grid search
C = np.logspace(-3, 2, 6)
param_grid = {'C': C}
skfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)

grid_search_lr = GridSearchCV(LogisticRegression(solver="liblinear"), param_grid, cv=skfold,
                           scoring='roc_auc', iid=False,
                           n_jobs=-1, return_train_score=True)

grid_search_lr.fit(X_trainval_selected, y_trainval)

print('Best params: {}'.format(grid_search_lr.best_params_))

clf_lr = LogisticRegression(C=100, solver="liblinear")
clf_lr.fit(X_trainval_scaled_selected, y_trainval) # lr model with the best params


#############################################################################
# ANN (Artificial Neural Network) - empirically searched with early stopping

X_train_selected, X_valid_selected, y_train_selected, y_valid_selected = train_test_split(X_trainval_selected, y_trainval,
                                                                                          test_size = 0.2, stratify=y_trainval,
                                                                                          random_state=0)

input_col_num = X_test_selected.shape[1]

def make_model(input_dim):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(input_dim*5, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation = 'relu', input_shape = (input_dim,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(rate = 0.1))

    model.add(tf.keras.layers.Dense(input_dim*5, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation = 'relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(rate = 0.1))
    
    model.add(tf.keras.layers.Dense(input_dim*5, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation = 'relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(rate = 0.1))

    model.add(tf.keras.layers.Dense(input_dim*3, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation = 'relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(rate = 0.1))
    
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

    model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

## early stopping
def find_best_epoch(model, X_train_scaled, y_train, X_valid_scaled, y_valid):
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 100)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose = 1,
                                  patience=5, min_lr=1e-15)

    callbacks = [reduce_lr, early_stopping]

    try_epoch = 1000

    history = model.fit(X_train_scaled,
                      y_train,
                      epochs=try_epoch,
                      batch_size=128,
                      validation_data=(X_valid_scaled, y_valid),
                      callbacks=callbacks)
    return (np.argmin(history.history['val_loss']) + 1, history)

model = make_model(input_col_num)

best_epoch, train_val_history = find_best_epoch(model, X_train_selected, y_train_selected,
                                                X_valid_selected, y_valid_selected)

## retrain the model with trainval datasets

model = make_model(input_col_num)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, verbose = 1,
                                  patience=100, min_lr=1e-15)

save_best = tf.keras.callbacks.ModelCheckpoint('ANN_model_recat.h5', monitor = 'loss', verbose = 1,
                                               save_bset_only = True, save_weights_only=False, mode = 'min')

callbacks = [reduce_lr, save_best]

ANN_model = model.fit(X_trainval_selected, y_trainval,
                    epochs=best_epoch, batch_size=128, callbacks = callbacks)

