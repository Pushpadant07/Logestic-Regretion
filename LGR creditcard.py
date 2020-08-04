import pandas as pd
import seaborn as sb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

# loading claimants data 
credit= pd.read_csv("D:\\ExcelR Data\\Assignments\\Logestic Regretion\\creditcard.csv")
credit.drop(["SlNo"],inplace=True,axis = 1)
# Credt= credt.iloc[:,1:] To Drop The Column we can use this also

Le = preprocessing.LabelEncoder() ##Label encoder() using for levels of categorical Data into numerical values
credit['Card'] = Le.fit_transform(credit['card'])
credit= credit.drop('card',axis = 1)
credit['Owner'] = Le.fit_transform(credit['owner'])
credit= credit.drop('owner',axis = 1)
credit['Selfemp'] = Le.fit_transform(credit['selfemp'])
credit= credit.drop('selfemp',axis = 1)
# Getting the barplot for the categorical columns 
#EDA Part
#cat_column = [Card,Owner,Selfemp,majorcards] 
#cont_column =[reports,age,income,share,expenditure,dependents,months,active]
sb.countplot(x="Card",data=credit)
pd.crosstab(credit.Card,credit.majorcards).plot(kind="bar")
sb.countplot(x="Owner",data=credit)
pd.crosstab(credit.Owner,credit.majorcards).plot(kind="bar")
sb.countplot(x="Selfemp",data=credit)
pd.crosstab(credit.Selfemp,credit.majorcards).plot(kind="bar")

sb.countplot(x="majorcards",data=credit)
# Data Distribution - Boxplot of continuous variables wrt to each category of categorical columns
sb.boxplot(data = credit,orient = "v")
sb.boxplot(x="Card",y="reports",data=credit,palette = "hls")
sb.boxplot(x="Card",y="age",data=credit,palette = "hls")
sb.boxplot(x="Card",y="income",data=credit,palette = "hls")
sb.boxplot(x="Card",y="share",data=credit,palette = "hls")
sb.boxplot(x="Card",y="expenditure",data=credit,palette = "hls")
sb.boxplot(x="Card",y="dependents",data=credit,palette = "hls")
sb.boxplot(x="Card",y="months",data=credit,palette = "hls")
sb.boxplot(x="Card",y="active",data=credit,palette = "hls")

credit.isnull().sum() #All the columns having no Null values

# No Null values so noe go for the Model Bulding

# Model building 
credit.shape
X = credit.iloc[:,[0,1,2,3,4,5,6,7,8,10,11]]
Y = credit.iloc[:,9]
classifier = LogisticRegression()
classifier.fit(X,Y)

classifier.coef_ # coefficients of features 
classifier.predict_proba (X) # Probability values 

y_pred = classifier.predict(X)
credit["y_pred"] = y_pred
y_prob = pd.DataFrame(classifier.predict_proba(X.iloc[:,:]))
new_df = pd.concat([credit,y_prob],axis=1)
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

