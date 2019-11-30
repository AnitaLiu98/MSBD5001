# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import learning_curve, validation_curve, GridSearchCV # Model evaluation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split # Model evaluation
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler,MinMaxScaler # Preprocessing
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor  # Ensemble methods



df=pd.read_csv('../dataprocessiing/processed data/prefeatures_dropold.csv')
print(df.head())
df_pre=pd.read_csv('../dataprocessiing/processed data/test_feature.csv')
slctdfeature=pd.read_csv('../dataprocessiing/processed data/slctdfeature .csv')
###################################################################################
##########    traindata and testdata################
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
#cv_params = {'n_estimators': range(25,35,1)}
#cv_params = {'max_depth':range(3,30,2)}#, }
#cv_params = {'min_samples_split':range(5,100,10)}
#cv_params = {'min_samples_leaf':range(5,100,10)}
#cv_params = {'min_samples_split':range(3,20,2), 'min_samples_leaf':range(3,15,2)}
cv_params ={'random_state':range(0,100,10)}
other_params = {'n_estimators' : 26, 'oob_score' : 'TRUE', 'n_jobs' : -1,'random_state' :20,'min_samples_split':15,
                'min_samples_leaf':5,'max_features':'sqrt','max_depth':5}

model = RandomForestRegressor(**other_params)
optimized_rf = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_absolute_error', cv=20, verbose=1, n_jobs=4)
#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'
print('optimized_rf:',optimized_rf)
optimized_rf.fit(X, y)
means = optimized_rf.cv_results_['mean_test_score']
stds = optimized_rf.cv_results_['std_test_score']
params = optimized_rf.cv_results_['params']
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

opt_params = {'n_estimators' : 29, 'oob_score' : 'TRUE', 'n_jobs' : -1,'random_state' :40,'min_samples_split':15,
                'min_samples_leaf':5,'max_features':'sqrt','max_depth':5}
model = RandomForestRegressor(**opt_params)
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
df_raw=pd.read_csv('../rawdata/test.csv')
resultdf = pd.DataFrame(pred)
resultdf.columns = ['playtime_forever']
for i in range(0,resultdf.shape[0]):
    if resultdf.loc[i,'playtime_forever'] <0:
        resultdf.loc[i,'playtime_forever'] = 0
    elif resultdf.loc[i, 'playtime_forever'] < 1.2:
        resultdf.loc[i, 'playtime_forever'] = 0
    elif resultdf.loc[i, 'playtime_forever'] > 4 and resultdf.loc[i, 'playtime_forever'] <8:
        resultdf.loc[i, 'playtime_forever'] = resultdf.loc[i, 'playtime_forever']*1.5
    elif resultdf.loc[i, 'playtime_forever'] >= 8 and resultdf.loc[i, 'playtime_forever'] <10:
        resultdf.loc[i, 'playtime_forever'] = resultdf.loc[i, 'playtime_forever']*2
    elif resultdf.loc[i, 'playtime_forever'] >=10 and resultdf.loc[i, 'playtime_forever'] <12 :
        resultdf.loc[i, 'playtime_forever'] = resultdf.loc[i, 'playtime_forever']*2.3
    elif resultdf.loc[i, 'playtime_forever'] >= 12 and resultdf.loc[i, 'playtime_forever'] <13 :
        resultdf.loc[i, 'playtime_forever'] = resultdf.loc[i, 'playtime_forever']*3
    elif resultdf.loc[i, 'playtime_forever'] >= 13 and resultdf.loc[i, 'playtime_forever'] <14 :
        resultdf.loc[i, 'playtime_forever'] = resultdf.loc[i, 'playtime_forever']*5.5
    elif resultdf.loc[i, 'playtime_forever'] >= 14:
        resultdf.loc[i, 'playtime_forever'] = resultdf.loc[i, 'playtime_forever']*7.5

resultdf = pd.concat([df_raw['id'],resultdf], axis=1, join='outer', join_axes=None, ignore_index=False)

resultdf.to_csv('./test.csv',index=False)





predt = optimized_rf.predict(X)
print(predt)
#predt = np.exp(predt)-1

