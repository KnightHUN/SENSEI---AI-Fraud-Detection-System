import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report
from joblib import dump

dataset = 'Dataset.csv'

df = pd.read_csv(dataset)

x = df.drop('Class' , axis = 1 )
y = df['Class']

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state=42
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

labeled_model = LogisticRegression()
labeled_model.fit(x_train,y_train)
model_name_1 = "LogisticRegression_model.joblib"
dump(labeled_model,model_name_1)
print("LogisticRegression model saved successfully") 


cluster_model = RandomForestClassifier()
cluster_model.fit(x_train,y_train)
model_name_2 = "RandomForestClassifier_model.joblib"
dump(cluster_model,model_name_2)
print("RandomForestClassifier model saved successfully")

lr_preds = labeled_model.predict(x_test)
rf_preds = cluster_model.predict(x_test)

print("\nLogistic Regression Performance:")
print("Accuracy:", accuracy_score(y_test, lr_preds))
print("Classification Report:")
print(classification_report(y_test, lr_preds))

print("\nRandom Forest Classifier Performance:")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("Classification Report:")
print(classification_report(y_test, rf_preds))