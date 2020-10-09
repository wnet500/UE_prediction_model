import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier


#############################################################################
'''
Here, the datasets we used in the research paper can not be released
for personal information protection.

Instead, you can identify the structure of datasets and their examples. 
please refer to "features_details.csv" and "UE_sample_data.csv" in the datasets folder
'''

# Load dataset
Dev_set = pd.read_csv('Development_sets.csv')
X_trainval = Dev_set.drop("Outcome", axis=1)
y_trainval = Dev_set['Outcome']

Val_set = pd.read_csv('test.csv')
X_test = Val_set.drop("Outcome", axis=1)
y_test = Val_set['Outcome']


# Normalization for numerical features
numerical_featurs =  pd.read_csv('../datasets/features_details.csv')
numerical_featurs = numerical_featurs[numerical_featurs['type'] == 'numerical']['features']

scaler = StandardScaler()
scaler.fit(X_trainval[:, numerical_featurs])

X_trainval.loc[:, numerical_featurs] = scaler.transform(X_trainval.loc[:, numerical_featurs])
X_test.loc[:, numerical_featurs] = scaler.transform(X_test.loc[:, numerical_featurs])


#############################################################################
# Feature selection using rfe cv
skfold = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
rfecv = RFECV(estimator=RandomForestClassifier(n_estimators=500, n_jobs = -1, random_state=0),
              step=1,
              cv=skfold,
              scoring='roc_auc',
              n_jobs = -1)
rfecv.fit(X_trainval, y_trainval)

X_trainval_selected = X_trainval.iloc[:, rfecv.get_support(indices=True)]
X_test_selected = X_test.iloc[:, rfecv.get_support(indices=True)]


# Feature importance analysis using random forest
rf_rfe = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
rf_rfe.fit(X_trainval, y_trainval)

rfe_feature_importances = pd.DataFrame(rf_rfe.feature_importances_,
                                      index = X_trainval.columns,
                                      columns=['importance']).sort_values('importance', ascending=False)

# display selected features and their feature importance
rfe_feature_importances.loc[X_trainval.columns[rfecv.get_support(indices=False)], :].sort_values('importance', ascending=False) 
