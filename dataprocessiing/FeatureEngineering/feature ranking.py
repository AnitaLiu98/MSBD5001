# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler # Preprocessing
from sklearn.linear_model import Lasso, Ridge, RandomizedLasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor # Ensemble methods
from sklearn.feature_selection import RFE, f_regression




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
Y=df.playtime_forever.values
#y= np.log(y+1)

colnames=df.iloc[:,1:].columns
print(X[0:5,0:10])
'''# Spliting X and y into train and test version
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state=3)'''

###################################################################################
##########feature selection################
###################################################################################



# Define dictionary to store our rankings
ranks = {}
# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

# Finally let's run our Selection Stability method with Randomized Lasso
rlasso = RandomizedLasso(alpha=0.04)
rlasso.fit(X, Y)
ranks["rlasso/Stability"] = ranking(np.abs(rlasso.scores_), colnames)
print('finished')


###################################################################################
##########feature selection################
###################################################################################
# Construct our Linear Regression model
lr = LinearRegression(normalize=True)
lr.fit(X,Y)
#stop the search when only the last feature is left
rfe = RFE(lr, n_features_to_select=1, verbose =3 )
rfe.fit(X,Y)
ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)





###################################################################################
##########feature selection################
###################################################################################
# Using Linear Regression
lr = LinearRegression(normalize=True)
lr.fit(X,Y)
ranks["LinReg"] = ranking(np.abs(lr.coef_), colnames)

# Using Ridge
ridge = Ridge(alpha = 7)
ridge.fit(X,Y)
ranks['Ridge'] = ranking(np.abs(ridge.coef_), colnames)

# Using Lasso
lasso = Lasso(alpha=.05)
lasso.fit(X, Y)
ranks["Lasso"] = ranking(np.abs(lasso.coef_), colnames)




rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)
rf.fit(X,Y)
ranks["RF"] = ranking(rf.feature_importances_, colnames)

# Create empty dictionary to store the mean value calculated from all the scores
r = {}
for name in colnames:
    r[name] = round(np.mean([ranks[method][name]
                             for method in ranks.keys()]), 2)

methods = sorted(ranks.keys())
ranks["Mean"] = r
methods.append("Mean")

print("\t%s" % "\t".join(methods))
for name in colnames:
    print("%s\t%s" % (name, "\t".join(map(str,
                                          [ranks[method][name] for method in methods]))))


# Put the mean scores into a Pandas dataframe
meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])

# Sort the dataframe
meanplot = meanplot.sort_values('Mean Ranking', ascending=False)
# Let's plot the ranking of the features
sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar",
               size=14, aspect=1.9, palette='coolwarm')



slctdfeature = meanplot[0:150]
slctdfeature .to_csv('../../dataprocessiing/processed data/slctdfeature .csv',index=False)

