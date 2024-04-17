import streamlit as st 
import pandas as pd 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

data = sns.load_dataset("iris")

X = data.drop(['species'],axis=1)
y = data['species']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

tr = DecisionTreeClassifier()
tr.fit(X_train,y_train)

with open("tree.pkl","wb") as f:
    pickle.dump(tr,f)