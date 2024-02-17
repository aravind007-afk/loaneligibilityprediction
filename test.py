import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
df = pd.read_csv("loan_approval_dataset.csv")
df.drop('loan_id',axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df[' education'] = le.fit_transform(df[' education'])
df[' self_employed'] = le.fit_transform(df[' self_employed'])
df[' loan_status'] = le.fit_transform(df[' loan_status'])
df.insert(2,' no_of_dependents',df.pop(' no_of_dependents'))
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1,shuffle=True)

xtrrf=xtrain
ytrrf=ytrain
from sklearn.ensemble import RandomForestClassifier
rfclass=RandomForestClassifier(n_estimators=75,max_features=None,criterion='log_loss')
# rfparams={
#     'criterion':['gini','entropy','log_loss'],
#     'max_features':['sqrt','log2',None],
#     'n_estimators':[25,50,75]
# }
# rfdt=GridSearchCV(estimator=rfclass,param_grid=rfparams,scoring='accuracy',cv=10,n_jobs=-1)
rfclass.fit(xtrrf,ytrrf)

# from sklearn.metrics import accuracy_score
# y_pred=rfclass.predict(xtest)
# accuracy_score(y_pred,ytest)
pickle.dump(rfclass,open('model.pkl','wb'))

