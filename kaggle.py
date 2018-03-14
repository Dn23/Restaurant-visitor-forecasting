# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 23:01:17 2018

@author: dwark
"""

import numpy as np
import pandas as pd
import datetime as dt


start_time = dt.datetime.now()
print("Started at ", start_time)

data = {
    'tra': pd.read_csv('air_visit_data.csv'),
    'as': pd.read_csv('air_store_info.csv'),
    'hs': pd.read_csv('hpg_store_info.csv'),
    'ar': pd.read_csv('air_reserve.csv'),
    'hr': pd.read_csv('hpg_reserve.csv'),
    'id': pd.read_csv('store_id_relation.csv'),
    'tes': pd.read_csv('sample_submission.csv'),
    'hol': pd.read_csv('date_info.csv').rename(columns={'calendar_date': 'visit_date'})
}
data['tra']['dow']=pd.to_datetime(data['tra']['visit_date']).dt.dayofweek
data['tra']=pd.merge(data['tra'],data['as'],how='left',on=['air_store_id'])
df=data['as'].groupby(['air_area_name','air_genre_name'],as_index=False)[['air_store_id']].count()
df=df.rename(columns={'air_store_id':'No_of_competitiors'})
data['tra']=pd.merge(data['tra'],df,how='left',on=['air_area_name','air_genre_name'])
data['tra']['year']=pd.to_datetime(data['tra']['visit_date']).dt.year
data['tra']['month']=pd.to_datetime(data['tra']['visit_date']).dt.month
data['tes']['visit_date']=data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))
data['tes']['dow'] = pd.to_datetime(data['tes']['visit_date']).dt.dayofweek
data['tes']=pd.merge(data['tes'],data['as'],how='left',on=['air_store_id'])
data['tes']=pd.merge(data['tes'],df,how='left',on=['air_area_name','air_genre_name'])
data['tes']['year']=pd.to_datetime(data['tes']['visit_date']).dt.year
data['tes']['month']=pd.to_datetime(data['tes']['visit_date']).dt.month
df_dow=data['tra'].groupby(['air_store_id','dow'],as_index=False).agg({'visitors':['min','max','std','mean','median']})
df_dow.columns=[''.join(col).strip() for col in df_dow.columns.values]    
unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i] * len(unique_stores)}) for i in range(7)],
                   axis=0, ignore_index=True).reset_index(drop=True)
stores=pd.merge(stores,df_dow,how='left',on=['air_store_id','dow'])
data['hol']=data['hol'].drop(['day_of_week'],axis=1)
data['tra'] = pd.merge(data['tra'], stores, how='left', on=['air_store_id', 'dow'])
data['tes'] = pd.merge(data['tes'], stores, how='left', on=['air_store_id', 'dow'])
data['tra'] = pd.merge(data['tra'], data['hol'], how='left', on='visit_date')
data['tes'] = pd.merge(data['tes'], data['hol'], how='left', on='visit_date')

#impute the null values
df1=data['tra'].groupby(['air_area_name','air_genre_name','dow']).transform(lambda x : x.fillna(x.median()))
df1[['air_store_id','air_genre_name','air_area_name','visit_date','dow']]=data['tra'][['air_store_id','air_genre_name','air_area_name','visit_date','dow']]
df2=df1.groupby(['air_area_name','dow']).transform(lambda x : x.fillna(x.median()))
df2[['air_store_id','air_genre_name','air_area_name','visit_date','dow']]=data['tra'][['air_store_id','air_genre_name','air_area_name','visit_date','dow']]
data['tra']=df2

df1=data['tes'].groupby(['air_area_name','air_genre_name','dow']).transform(lambda x : x.fillna(x.median()))
df1[['id','air_store_id','air_genre_name','air_area_name','visit_date','dow']]=data['tes'][['id','air_store_id','air_genre_name','air_area_name','visit_date','dow']]
df2=df1.groupby(['air_area_name','dow']).transform(lambda x : x.fillna(x.median()))
df2[['id','air_store_id','air_genre_name','air_area_name','visit_date','dow']]=data['tes'][['id','air_store_id','air_genre_name','air_area_name','visit_date','dow']]
data['tes']=df2
data['tra']['visitors'].hist(bins=50)
import seaborn as sns
sns.boxplot(data=data['tra'],y='visitors')
sns.pairplot(data=data['tra'][['visitors','visitorsmin','visitorsmax','visitorsmean','visitorsmedian','visitorsstd']])
data['tra']['visitors_log']=np.log(data['tra']['visitors'])
data['tra']['visitors_log'].hist()
dummy=pd.get_dummies(data['tra'][['air_genre_name','air_area_name']],columns=['air_genre_name','air_area_name'])
data['tra']=pd.concat([data['tra'],dummy],axis=1)
dummy=pd.get_dummies(data['tes'][['air_area_name','air_genre_name']],columns=['air_area_name','air_genre_name'])
data['tes']=pd.concat([data['tes'],dummy],axis=1)
X=data['tra'].drop(['air_area_name','air_genre_name','air_store_id','visit_date','visitors','visitors_log'],axis=1)
y=data['tra']['visitors_log']
X.columns
#running linear regression model
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
from sklearn.model_selection import cross_val_score
scores=cross_val_score(lm,X,y)
scores.mean()
from sklearn.linear_model import Lasso
"""alphas=[0.0005,0.001,0.002]
scores=[]
for alpha in alphas :
    lasso=Lasso(alpha=alpha)
    scores.append(cross_val_score(lasso,X,y).mean())
import matplotlib.pyplot as plt
plt.plot(alphas,scores)
"""
lasso=Lasso(alpha=0.002)
lasso=lasso.fit(X,y)
lasso.coef_
print('Non-zero features: {}'
     .format(np.sum(lasso.coef_ != 0)))
cols=[]
for e in list(zip(list(X), lasso.coef_)) :
    if e[1] != 0:
        print('\t{}, {:.3f}'.format(e[0], e[1]))
        cols.append(e[0])
X=X[cols]
lasso=Lasso(alpha=0.002)
lasso=lasso.fit(X,y)
cross_val_score(lasso,X,y).mean()
X_test=data['tes'][cols]
sub=pd.DataFrame()
sub['id']=data['tes']['id']
sub['visitors']=np.exp(lasso.predict(X_test))
sub.to_csv("submission2.csv",index=False)

from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
from sklearn.pipeline import make_pipeline
pipe=make_pipeline(MinMaxScaler(),LinearRegression())
pipe1=make_pipeline(StandardScaler(),LinearRegression())
pipe2=make_pipeline(Normalizer(),LinearRegression())
cross_val_score(pipe,X,y).mean()

#more feature engineering
X['vistorsmin_log']=np.log1p(X['visitorsmin'])
X['vistorsmax_log']=np.log1p(X['visitorsmax'])
X['vistorsmean_log']=np.log1p(X['visitorsmean'])
X['vistorsstd_log']=np.log1p(X['visitorsstd'])
X=X.drop(['visitorsmin','visitorsmean','visitorsmax','visitorsstd'],axis=1)

lm=LinearRegression()
from sklearn.model_selection import cross_val_score
scores=cross_val_score(lm,X,y)
scores.mean()

pipe=make_pipeline(MinMaxScaler(),LinearRegression())
pipe1=make_pipeline(StandardScaler(),LinearRegression())
pipe2=make_pipeline(Normalizer(),LinearRegression())
print(cross_val_score(pipe,X,y).mean())
print(cross_val_score(pipe1,X,y).mean())
print(cross_val_score(pipe2,X,y).mean())

X_test['vistorsmin_log']=np.log1p(X_test['visitorsmin'])
X_test['vistorsmax_log']=np.log1p(X_test['visitorsmax'])
X_test['vistorsmean_log']=np.log1p(X_test['visitorsmean'])
X_test['vistorsstd_log']=np.log1p(X_test['visitorsstd'])
X_test=X_test.drop(['visitorsmin','visitorsmean','visitorsmax','visitorsstd'],axis=1)
lm=LinearRegression()
lm.fit(X,y)
sub=pd.DataFrame()
sub['id']=data['tes']['id']
sub['visitors']=np.exp(lm.predict(X_test))
sub.to_csv("submission4.csv",index=False)

X_train=pd.concat([X,X_test],axis=0)

dummy=pd.get_dummies(X_train[['year','month']],columns=['year','month'])
X_train=pd.concat([X_train,dummy],axis=1)
X_train=X_train.reset_index(drop=True)
X=X_train.iloc[0:252108,:]
X_test=X_train.iloc[252108 :,:]
X['competition_log']=np.log1p(X['No_of_competitiors'])
X['latitude_log']=np.log1p(X['latitude'])
X['longitude_log']=np.log1p(X['longitude'])
X_test['competition_log']=np.log1p(X_test['No_of_competitiors'])
X_test['latitude_log']=np.log1p(X_test['latitude'])
X_test['longitude_log']=np.log1p(X_test['longitude'])
X=X.drop(['No_of_competitiors','latitude','longitude'],axis=1)
X_test=X_test.drop(['No_of_competitiors','latitude','longitude'],axis=1)
alphas=[0.0005,0.001,0.002,0.1]
scores=[]
for alpha in alphas :
    lasso=Lasso(alpha=alpha)
    scores.append(cross_val_score(lasso,X,y).mean())
import matplotlib.pyplot as plt
plt.plot(alphas,scores)

lm=LinearRegression()
lm.fit(X,y)
scores=cross_val_score(lm,X,y)
scores.mean()

pipe=make_pipeline(MinMaxScaler(),LinearRegression())
pipe1=make_pipeline(StandardScaler(),LinearRegression())
pipe2=make_pipeline(Normalizer(),LinearRegression())
print(cross_val_score(pipe,X,y).mean())
print(cross_val_score(pipe1,X,y).mean())
print(cross_val_score(pipe2,X,y).mean())

scaler=Normalizer()
X=scaler.fit_transform(X)
X_test=scaler.transform(X_test)
lm=LinearRegression()
lm.fit(X,y)

sub=pd.DataFrame()
sub['id']=data['tes']['id']
sub['visitors']=np.exp(lm.predict(X_test))
sub.to_csv("submission5.csv",index=False)

params = {'n_estimators': 4000, # change to 4000 to achieve LB 0.511 - it runs too long in the kaggle kernel mode
        'max_depth': 5,
        'min_samples_split': 200, 
        'min_samples_leaf': 50,
        'learning_rate': 0.005,
        'max_features':  9,
        'subsample': 0.8,
        'loss': 'ls'}
from sklearn.ensemble import GradientBoostingRegressor
clf = GradientBoostingRegressor(**params)
pipe=make_pipeline(MinMaxScaler(),clf)
pipe1=make_pipeline(StandardScaler(),clf)
pipe2=make_pipeline(Normalizer(),clf)
print(cross_val_score(pipe,X,y).mean())
print(cross_val_score(pipe1,X,y).mean())
print(cross_val_score(pipe2,X,y).mean())

scores=cross_val_score(clf,X,y)
scores.mean()

scaler=Normalizer()
X=scaler.fit_transform(X)
X_test=scaler.transform(X_test)
clf.fit(X,y)
sub=pd.DataFrame()
sub['id']=data['tes']['id']
sub['visitors']=np.exp(clf.predict(X_test))
sub.to_csv("submission6.csv",index=False)