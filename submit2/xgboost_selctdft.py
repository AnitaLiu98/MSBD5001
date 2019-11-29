# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import GridSearchCV # Model evaluation
from xgboost import XGBRegressor # XGBoost
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import  train_test_split # Model evaluation
from sklearn.preprocessing import  StandardScaler,MinMaxScaler # Preprocessing



df=pd.read_csv('../dataprocessiing/processed data/prefeatures_dropold.csv')
print(df.head())
df_pre=pd.read_csv('../dataprocessiing/processed data/test_feature.csv')
slctdfeature=pd.read_csv('../dataprocessiing/processed data/slctdfeature .csv')
###################################################################################
##########                  traindata and test data                ################
###################################################################################

X = df.drop(['playtime_forever'],axis=1)
X = X[list(slctdfeature['Feature'])]
print(X.columns)
X = X.as_matrix()
y =df.playtime_forever.values
#y = np.log(y+1)
colnames=df.iloc[:,1:].columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state=3)


X_pre = df_pre
X_pre = X_pre[list(slctdfeature['Feature'])]
X_pre = X_pre.as_matrix()







###################################################################################
###################################################################################
##########                         CROSS VALIDATION               #################
###################################################################################
cv_params = {'n_estimators': range(1,20,2)}
#cv_params = {'max_depth': range(3,30,2)}
#cv_params = {'min_child_weight':range(1,6,1)}
#cv_params ={'gamma':[i/10.0 for i in range(0,5)] }
#cv_params ={'reg_lambda':[0.1, 0.5, 1, 5, 10,100,1000]}
#cv_params ={'reg_lambda':range(10,30,2)}
#cv_params ={'subsample': [0.8,0.4,0.6,0.2]}
#cv_params ={'learning_rate':[0.0001,0.001,0.01,0.1]}
other_params = {'learning_rate': 0.1, 'n_estimators': 3, 'max_depth': 19, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.2, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 14,'objective':'reg:squarederror'}

model = XGBRegressor(**other_params)
optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_absolute_error', cv=20, verbose=1, n_jobs=4)
#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'
optimized_GBM.fit(X, y)
means = optimized_GBM.cv_results_['mean_test_score']
stds = optimized_GBM.cv_results_['std_test_score']
params = optimized_GBM.cv_results_['params']
for mean, std, params in zip(means, stds, params):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))



###################################################################################
###################################################################################
##########                         RREDICTION              #################
###################################################################################



scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_pre = scaler.transform(X_pre)

opt_params = {'learning_rate': 0.1, 'n_estimators': 3, 'max_depth': 19, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.2, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 14,'objective':'reg:squarederror'}
model = XGBRegressor(**opt_params)
model.fit(X, y)

mse = mean_squared_error(model.predict(X), y)
print('mse',mse)

pred = model.predict(X_pre)
print(pred)




'''sns.set(style="white", font_scale=1)
plt.figure(figsize=(6.5,5))
print(newX_train)
plt.scatter(newX_train,y_train,color='darkgreen',label="Data", alpha=0.001)
plt.scatter(newX_pre,pred,color="red",label="Predicted Regression Line")
plt.xlabel("total_positive_reviews", fontsize=15)
plt.ylabel("playtime_foreve", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend()
plt.show()'''
#pred = np.exp(pred)-1
df_raw=pd.read_csv('/Users/liuxueling/PycharmProjects/python3pj/5001personal/rawdata/test.csv')
resultdf = pd.DataFrame(pred)
resultdf.columns = ['playtime_forever']
for i in range(0,resultdf.shape[0]):
    if resultdf.loc[i,'playtime_forever'] <0:
        resultdf.loc[i,'playtime_forever'] = 0

resultdf = pd.concat([df_raw['id'],resultdf], axis=1, join='outer', join_axes=None, ignore_index=False)

resultdf.to_csv('./test.csv',index=False)





predt = optimized_GBM.predict(X)
print(predt)
#predt = np.exp(predt)-1

