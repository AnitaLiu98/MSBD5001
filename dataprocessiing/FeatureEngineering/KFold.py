# -*- coding: utf-8 -*-
import pandas as pd
from xgboost import XGBRegressor, plot_importance # XGBoost
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, train_test_split # Model evaluation
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler,MinMaxScaler # Preprocessing
from sklearn.linear_model import Lasso, Ridge, ElasticNet, RANSACRegressor, SGDRegressor, HuberRegressor, BayesianRidge # Linear models
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor  # Ensemble methods
from sklearn.svm import SVR, SVC, LinearSVC  # Support Vector Regression
from sklearn.tree import DecisionTreeRegressor # Decision Tree Regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline # Streaming pipelines
from sklearn.decomposition import KernelPCA, PCA # Dimensionality reduction





evaluation = pd.DataFrame({'Model': [],
                           'Details':[],
                           'Root Mean Squared Error (RMSE)':[],
                           'R-squared (training)':[],
                           'Adjusted R-squared (training)':[],
                           'R-squared (test)':[],
                           'Adjusted R-squared (test)':[],
                           '5-Fold Cross Validation':[]})




df=pd.read_csv('../../dataprocessiing/processed data/prefeatures_dropold.csv')

print(df.head())

# Looking for nulls
print(df.isnull().any())
# Inspecting type
print(df.dtypes)


#Pearson Correlation Matrix
'''features = ['playtime_forever','is_free','price','total_positive_reviews','total_negative_reviews','eSports_tags',
       'date_diff', 'pur_num_ym']
df2=df.iloc[:,46:60]
df2['playtime_forever']=df['playtime_forever']
mask = np.zeros_like(df2.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(16, 16))
plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(df2.corr(),square=True,cmap="BuGn", #"BuGn_r" to reverse
            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});
plt.show()'''

###################################################################################
##########    traindata and testdata################
###################################################################################
X = df.drop(['playtime_forever'],axis=1).as_matrix()
y=df.playtime_forever.values
#y= np.log(y+1)
print(y)
colnames=df.iloc[:,1:].columns
print(X[0:5,0:10])
# Spliting X and y into train and test version
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state=3)

###################################################################################
'''##########feature selection################
###################################################################################
def ranking(ranks, names, order=1):
    minmax = StandardScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))


model = XGBRegressor()
model.fit(X_train, y_train)
ranks=pd.DataFrame(columns=['xgbr'])

# make predictions for test data and evaluate
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
mse = MSE(y_test, predictions)
print("mse: %.2f" %(mse))
print('y:\n',y_test)
print('pre:\n',predictions)
#select features using threshold
print('impor',model.feature_importances_)
thresh = [5 * 10**(-3)]
for c in thresh:
    selection = SelectFromModel(model, threshold=c, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBRegressor(objective ='reg:squarederror')
    selection_model.fit(select_X_train, y_train)
    # eval mode
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    predictions = [round(value) for value in y_pred]
    mse = MSE(y_test, predictions)
    print("Threshold = %.6f, n = %d, MSE: %.2f" % (c, select_X_train.shape[1], mse))
    #print (select_X_train)
    ranks["xgbr"] = ranking(model.feature_importances_, colnames,order=-1)
    print ('ranks',ranks['xgbr'])


for name in colnames:
    print("%s\t%s" % (name, "\t".join(map(str,[ranks['xgbr'][name]]))))'''
###################################################################################
##########CROSS VALIDATION################
###################################################################################
pipelines = []
seed = 2

pipelines.append(
                ("Scaled_Ridge",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("Ridge", Ridge(random_state=seed, tol=10 ))
                      ]))
                )
pipelines.append(
                ("Scaled_Lasso",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("Lasso", Lasso(random_state=seed, tol=1))
                      ]))
                )
pipelines.append(
                ("Scaled_Elastic",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("Lasso", ElasticNet(random_state=seed))
                      ]))
                )

pipelines.append(
                ("Scaled_SVR",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("SVR",  SVR(kernel='linear', C=1e2, degree=5))
                 ])
                )
                )

pipelines.append(
                ("Scaled_RF_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("RF", RandomForestRegressor(random_state=seed))
                 ])
                )
                )

pipelines.append(
                ("Scaled_ET_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("ET", ExtraTreesRegressor(random_state=seed))
                 ])
                )
                )

pipelines.append(
                ("Scaled_BR_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("BR", BaggingRegressor(random_state=seed))
                 ])))

pipelines.append(
                ("Scaled_Hub-Reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("Hub-Reg", HuberRegressor())
                 ])))
pipelines.append(
                ("Scaled_BayRidge",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("BR", BayesianRidge())
                 ])))

pipelines.append(
                ("Scaled_XGB_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("XGBR", XGBRegressor(seed=seed))
                 ])))

pipelines.append(
                ("Scaled_DT_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("DT_reg", DecisionTreeRegressor())
                 ])))

pipelines.append(
                ("Scaled_KNN_reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("KNN_reg", KNeighborsRegressor())
                 ])))
#pipelines.append(
#                ("Scaled_ADA-Reg",
#                 Pipeline([
#                     ("Scaler", StandardScaler()),
#                     ("ADA-reg", AdaBoostRegressor())
#                 ])))

pipelines.append(
                ("Scaled_Gboost-Reg",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("GBoost-Reg", GradientBoostingRegressor())
                 ])))

pipelines.append(
                ("Scaled_RFR_PCA",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("PCA", PCA(n_components=5)),
                     ("XGB", RandomForestRegressor())
                 ])))

pipelines.append(
                ("Scaled_XGBR_PCA5",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("PCA", PCA(n_components=5)),
                     ("XGB", XGBRegressor())
                 ])))
pipelines.append(
                ("Scaled_XGBR_PCA20",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("PCA", PCA(n_components=20)),
                     ("XGB", XGBRegressor())
                 ])))
pipelines.append(
                ("Scaled_XGBR_PCA30",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("PCA", PCA(n_components=30)),
                     ("XGB", XGBRegressor())
                 ])))
pipelines.append(
                ("Scaled_RF_reg_PCA",
                 Pipeline([
                     ("Scaler", StandardScaler()),
                     ("PCA", PCA(n_components=5)),
                     ("RF", RandomForestRegressor(random_state=seed))
                 ])))
#'neg_mean_absolute_error', 'neg_mean_squared_error','r2'

scoring = 'neg_mean_squared_error'   #'r2'  'neg_mean_squared_error'
n_folds = 2

results, names = [], []




for name, model in pipelines:
    kfold = KFold(n_splits=n_folds, random_state=seed)
    cv_results = cross_val_score(model, X, y, cv=kfold,
                                 scoring=scoring, n_jobs=-1)
    names.append(name)
    results.append(cv_results)
    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure(figsize=(15, 6))
fig.suptitle('Algorithm Comparison', fontsize=22)
ax = fig.add_subplot(111)
sns.boxplot(x=names, y=results)
ax.set_xticklabels(names)
ax.set_xlabel("Algorithmn Name", fontsize=20)
ax.set_ylabel("R2 of Models", fontsize=18)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.show()



###################################################################################
'''
#simple regression
#get train and test
train_data,test_data = train_test_split(df,train_size = 0.6,random_state=3)

lr = linear_model.LinearRegression()
X_train = np.array(train_data['total_positive_reviews'], dtype=pd.Series).reshape(-1,1)
y=train_data['playtime_forever']
y= y.map(lambda x: np.log(x+1))
print(y)
y_train = np.array(y, dtype=pd.Series)
lr.fit(X_train,y_train)

X_test = np.array(test_data['total_positive_reviews'], dtype=pd.Series).reshape(-1,1)
y=test_data['playtime_forever']
y= y.map(lambda x: np.log(x+1))
print(y)
y_test = np.array(y, dtype=pd.Series)

#####################################################
pred = lr.predict(X_test)
rmsesm = float(format(np.sqrt(metrics.mean_squared_error(y_test,pred)),'.3f'))
rtrsm = float(format(lr.score(X_train, y_train),'.3f'))
rtesm = float(format(lr.score(X_test, y_test),'.3f'))
cv = float(format(cross_val_score(lr,df[['total_positive_reviews']],df['playtime_forever'],cv=5).mean(),'.3f'))

print ("Average Price for Test Data: {:.3f}".format(y_test.mean()))
print('Intercept: {}'.format(lr.intercept_))
print('Coefficient: {}'.format(lr.coef_))

r = evaluation.shape[0]
evaluation.loc[r] = ['Simple Linear Regression','-',rmsesm,rtrsm,'-',rtesm,'-',cv]
evaluation
sns.set(style="white", font_scale=1)
plt.figure(figsize=(6.5,5))
plt.scatter(X_test,y_test,color='darkgreen',label="Data", alpha=.1)
plt.plot(X_test,lr.predict(X_test),color="red",label="Predicted Regression Line")
plt.xlabel("total_positive_reviews", fontsize=15)
plt.ylabel("playtime_foreve", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend()

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.show()
'''




