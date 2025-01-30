#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')


# In[2]:


dataframe = pd.read_csv("PS_20174392719_1491204439457_log.csv")
dataframe.head()


# In[3]:


dataframe.info()


# In[4]:


dataframe.isnull().sum()


# In[5]:


dataframe = dataframe.drop(columns=['nameOrig', 'nameDest'])


# In[6]:


dataframe.head()


# In[7]:


labelEncoder = LabelEncoder()
dataframe['type'] = labelEncoder.fit_transform(dataframe['type'])


# In[8]:


dataframe.head()


# In[9]:


dataframe.info()


# In[10]:


dataframe.isnull().sum()


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, precision_recall_curve


# In[12]:


x_data = dataframe.drop(columns=['isFraud', 'isFlaggedFraud'])
y_data = dataframe['isFraud']


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)


# In[14]:


logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train, y_train)


# In[15]:


predict_y_logmodel = logmodel.predict(X_test)


# In[16]:


accuracy_logmodel = accuracy_score(y_test, predict_y_logmodel)
print(f"Accuracy of Logistic Regression Model: {accuracy_logmodel:.2f}")


# In[17]:


report_logmodel = classification_report(y_test, predict_y_logmodel)
print(report_logmodel)


# In[18]:


con_matrix_logmodel = metrics.confusion_matrix(y_test,predict_y_logmodel)
con_matrix_logmodel


# In[19]:


p = sns.heatmap(pd.DataFrame(con_matrix_logmodel), annot=True, annot_kws={"size": 18}, cmap="plasma" ,fmt='g')

plt.title('Confusion matrix of Logistic Regression', y=1.1, fontsize = 22)
plt.ylabel('Actual',fontsize = 15)
plt.xlabel('Predicted',fontsize = 15)

plt.show()


# In[20]:


predict_y_logmodel_proba = logmodel.predict_proba(X_test)[:, 1]


# In[21]:


roc_auc_logmodel = roc_auc_score(y_test, predict_y_logmodel_proba)
print(f"\nAUC-ROC of Logistic Regression: {roc_auc_logmodel:.2f}")


# In[22]:


fpr, tpr, thresholds = roc_curve(y_test, predict_y_logmodel_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', label=f"AUC = {roc_auc_logmodel:.2f}")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve of Logistic Regression")
plt.legend(loc="lower right")
plt.grid()
plt.show()


# In[23]:


ddataframe_conf_matrix = pd.DataFrame(
    con_matrix_logmodel,
    columns=["Predicted_Not_Fraud", "Predicted_Fraud"],
    index=["Actual_Not_Fraud", "Actual_Fraud"]
)
ddataframe_conf_matrix.to_csv("dataframe_confusion_matrix.csv")


# In[25]:


precision, recall, thresholds = precision_recall_curve(y_test, predict_y_logmodel_proba)
dataframe_pr = pd.DataFrame({
    "Thresholds": np.append(thresholds, 1),  
    "Precision": precision,
    "Recall": recall
})
dataframe_pr.to_csv("dataframe_precision_recall_data.csv", index=False)

