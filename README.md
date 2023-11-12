# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE AND OUTPUT:
```py
Developed By : Kanishka.V.S
Reg No : 212222230061
```
# DATA PREPROCESSING BEFORE FEATURE SELECTION:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/titanic_dataset.csv')
df.head()
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex-07/assets/113497357/7507a88a-e400-49e7-afb0-b39ed6307c06)
# CHECKING NULL VALUES:
```py
df.isnull().sum()
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex-07/assets/113497357/323d9542-4dc8-40a5-8de5-a8dcd3ffe2af)
# DROPPING UNWANTED DATAS:
```py
df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
df.head()
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex-07/assets/113497357/49ba9f2e-b856-44ca-a096-78755d1bee0e)
# DATA CLEANING:
```py
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex-07/assets/113497357/1f93dcf6-f518-41bb-81a1-2ae077a93849)

# REMOVING OUTLIERS:
### Before:
```py
plt.title("Dataset with outliers")
df.boxplot()
plt.show()
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex-07/assets/113497357/d0ba9fd1-e2c6-4c6c-b161-df7d4f2139a1)
### After:
```py
cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex-07/assets/113497357/974ec683-cfe4-4756-9842-e7456a3b27b5)
# Feature Selection:
```py
from sklearn.preprocessing import OrdinalEncoder
climate = ['C','S','Q']
en= OrdinalEncoder(categories = [climate])
df['Embarked']=en.fit_transform(df[["Embarked"]])
df.head()
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex-07/assets/113497357/d15e4884-10d5-444a-8899-7956a67f753c)
```py
from sklearn.preprocessing import OrdinalEncoder
gender = ['male','female']
en= OrdinalEncoder(categories = [gender])
df['Sex']=en.fit_transform(df[["Sex"]])
df.head()
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex-07/assets/113497357/33c50448-2049-4a99-959a-3059257a01e5)
```
py
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])
df.head()
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex-07/assets/113497357/6ea47920-0def-4650-b7b3-673cb36d413b)
```py
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()
df1["Survived"]=np.sqrt(df["Survived"])
df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])
df1["Sex"]=np.sqrt(df["Sex"])
df1["Age"]=df["Age"]
df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])
df1["Fare"],parameters=stats.yeojohnson(df["Fare"])
df1["Embarked"]=df["Embarked"]
df1.skew()
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex-07/assets/113497357/e25a8645-d7c9-4079-8ac5-a268fa9540b7)
``py
import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1) 
y = df1["Survived"] 
```
# FILTER METHOD:
```py
plt.figure(figsize=(7,6))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex-07/assets/113497357/b58bdb6d-13c4-4ee9-aff5-c5ca99644223)

# HIGHLY CORRELATED FEATURES WITH THE OUTPUT VARIABLE SURVIVED:
```py
cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex-07/assets/113497357/eb36808b-7890-47ea-8bb3-8d46b60f8a44)

# BACKWARD ELIMINATION:
```py
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex-07/assets/113497357/dc8816d2-f6d9-4578-b358-ac7b701b41b6)
# OPTIMUM NUMBER OF FEATURES THAT HAVE HIGH ACCURACY:
```py
nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,step=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex-07/assets/113497357/5075fda2-cf1d-416b-9394-92ff63f7df36)
# FINAL SET OF FEATURE:
```py
cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, step=2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex-07/assets/113497357/38a07d08-6d09-40d4-a2e5-3ce4478d7432)
# EMBEDDED METHOD:
```py
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()
```
![image](https://github.com/kanishka2305/ODD2023-Datascience-Ex-07/assets/113497357/67cf9df9-65ae-4ac3-96e2-37ca64fb6e67)
# Result:
Thus, the various feature selection techniques have been performed on a given dataset successfully.
