import pandas as pd
import seaborn as sb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

# loading claimants data 
bankdata = pd.read_csv("D:\\ExcelR Data\\Assignments\\Logestic Regretion\\Bank.csv")

bankdata.head()
#Droping Any column is not Required

Le = preprocessing.LabelEncoder() ##Label encoder() using for levels of categorical Data into numerical values
bankdata['Default'] = Le.fit_transform(bankdata['default'])
bankdata= bankdata.drop('default',axis = 1)
bankdata['Housing'] = Le.fit_transform(bankdata['housing'])
bankdata = bankdata.drop('housing',axis = 1)
bankdata['Loan'] = Le.fit_transform(bankdata['loan'])
bankdata= bankdata.drop('loan',axis = 1)
bankdata['Y'] = Le.fit_transform(bankdata['y'])
bankdata= bankdata.drop('y',axis = 1)
bankdata['Job'] = Le.fit_transform(bankdata['job'])
bankdata= bankdata.drop('job',axis = 1)
bankdata['Marital'] = Le.fit_transform(bankdata['marital'])
bankdata= bankdata.drop('marital',axis = 1)
bankdata['YEducation'] = Le.fit_transform(bankdata['education'])
bankdata= bankdata.drop('education',axis = 1)
bankdata['Contact'] = Le.fit_transform(bankdata['contact'])
bankdata= bankdata.drop('contact',axis = 1)
bankdata['Poutcome'] = Le.fit_transform(bankdata['poutcome'])
bankdata= bankdata.drop('poutcome',axis = 1)
bankdata['Month'] = Le.fit_transform(bankdata['month'])
bankdata= bankdata.drop('month',axis = 1)
# Getting the barplot for the categorical columns 
#EDA Part
#cat_column = [Default,Housing,Loan,Y,previous] 
#cont_column = [age,job,marital,education,balance,contact,day,month,duration campaign pdays,poutcome] 
sb.countplot(x="Default",data=bankdata)
pd.crosstab(bankdata.Default,bankdata.Y).plot(kind="bar")
sb.countplot(x="Housing",data=bankdata)
pd.crosstab(bankdata.Housing,bankdata.Y).plot(kind="bar")
sb.countplot(x="Loan",data=bankdata)
pd.crosstab(bankdata.Loan,bankdata.Y).plot(kind="bar")

sb.countplot(x="Y",data=bankdata)
# Data Distribution - Boxplot of continuous variables wrt to each category of categorical columns
sb.boxplot(data = bankdata,orient = "v")
sb.boxplot(x="Default",y="age",data=bankdata,palette = "hls")
sb.boxplot(x="Default",y="balance",data=bankdata,palette = "hls")
sb.boxplot(x="Default",y="day",data=bankdata,palette = "hls")
sb.boxplot(x="Default",y="duration",data=bankdata,palette = "hls")
sb.boxplot(x="Default",y="campaign",data=bankdata,palette = "hls")
sb.boxplot(x="Default",y="pdays",data=bankdata,palette = "hls")
sb.boxplot(x="Y",y="age",data=bankdata,palette = "hls")
sb.boxplot(x="Y",y="balance",data=bankdata,palette = "hls")
sb.boxplot(x="Y",y="day",data=bankdata,palette = "hls")
sb.boxplot(x="Y",y="duration",data=bankdata,palette = "hls")
sb.boxplot(x="Y",y="campaign",data=bankdata,palette = "hls")
sb.boxplot(x="Y",y="pdays",data=bankdata,palette = "hls")

# To get the count of null values in the data 

bankdata.isnull().sum() #All the columns having no Null values

# No Null values so noe go for the Model Bulding

# Model building 
bankdata.shape
X = bankdata.iloc[:,[0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16]]
Y = bankdata.iloc[:,10]
classifier = LogisticRegression()
classifier.fit(X,Y)

classifier.coef_ # coefficients of features 
classifier.predict_proba (X) # Probability values 

y_pred = classifier.predict(X)
bankdata["y_pred"] = y_pred
y_prob = pd.DataFrame(classifier.predict_proba(X.iloc[:,:]))
new_df = pd.concat([bankdata,y_prob],axis=1)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,y_pred)
print (confusion_matrix)

########### ROC curve ###########
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(Y, y_pred)

auc = roc_auc_score(Y, y_pred)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, color='orange', label='ROC')
